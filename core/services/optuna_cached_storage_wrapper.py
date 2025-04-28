import itertools
import logging
import time
from typing import Dict, List, Optional, Any, Union, Iterable
import sqlalchemy
import sqlalchemy.orm as sqlalchemy_orm
import optuna
import dataclasses
from dataclasses import dataclass
import contextlib
from optuna.storages._rdb import models
from optuna.storages._rdb.storage import _create_scoped_session
from optuna.trial import FrozenTrial

from core.utils.time_utils import perf_log_text

logger = logging.getLogger(__name__)

@dataclass
class StudyNamePrefixCache:
    """Cache of study information for a specific prefix."""
    prefix: str
    completed_trial_counts_by_study: Dict[str, int] = dataclasses.field(default_factory=dict)
    loaded: bool = False
    
    @property
    def study_names(self) -> List[str]:
        """Get study names from the keys of completed_trial_counts_by_study"""
        return list(self.completed_trial_counts_by_study.keys())


class OptunaCachedStorageWrapper:
    """
    A wrapper for Optuna's SQL storage to provide additional functionality for querying
    studies and trials. This class provides convenient methods for analyzing optimization results.
    """
    
    def __init__(self, storage: Union[str, optuna.storages.BaseStorage]):
        """
        Initialize the Optuna SQL storage wrapper.
        
        Args:
            storage: Either an existing Optuna storage object or a connection string.
        """
        if isinstance(storage, str):
            self.storage = optuna.storages.get_storage(storage)
        else:
            self.storage = storage
            
        if not hasattr(self.storage, '_backend'):
            raise ValueError("The provided storage must be a database storage with a _backend attribute")
        
        self._session = self.storage._backend.scoped_session
        self._engine = self.storage._backend.engine
        self.prefix_caches = {}
    
    @contextlib.contextmanager
    def _create_session(self):
        """Create a session that is committed and closed when exiting the context."""
        with _create_scoped_session(self._session) as session:
            yield session
    
    def get_or_create_prefix_cache(self, prefix: str) -> StudyNamePrefixCache:
        """Get or create and populate a prefix cache for the given prefix."""
        if prefix not in self.prefix_caches:
            cache = StudyNamePrefixCache(prefix=prefix)
            self.prefix_caches[prefix] = cache
            self.load_trial_counts_for_prefix(prefix)
        return self.prefix_caches[prefix]
    
    def load_trial_counts_for_prefix(self, prefix: str) -> None:
        """
        Load completed trial counts for studies matching a prefix.
        This is a synchronous operation that directly queries the database.
        """
        cache = self.prefix_caches.get(prefix)
        if not cache or cache.loaded:
            return
        
        try:
            # Direct synchronous call
            completed_counts = self._count_completed_trials_sync(prefix)
            cache.completed_trial_counts_by_study = completed_counts
            cache.loaded = True
        except Exception as e:
            logger.error(f"Error loading trial counts for prefix '{prefix}': {e}")
            # Mark as loaded even on error to prevent repeated attempts to load
            if cache:
                cache.loaded = True
                cache.completed_trial_counts_by_study = {}
    
    def _count_completed_trials_sync(self, prefix: str) -> Dict[str, int]:
        """Get completed trial counts for studies matching a prefix, including studies with zero completed trials."""
        with self._create_session() as session:
            # Single query using LEFT JOIN to get all studies and their completed trial counts
            query = (
                session.query(
                    models.StudyModel.study_name,
                    sqlalchemy.func.count(
                        sqlalchemy.case(
                            (models.TrialModel.state == optuna.trial.TrialState.COMPLETE, models.TrialModel.trial_id),
                            else_=None
                        )
                    ).label('completed_count')
                )
                .outerjoin(  # LEFT JOIN to include studies with no trials
                    models.TrialModel,
                    models.StudyModel.study_id == models.TrialModel.study_id
                )
                .filter(models.StudyModel.study_name.like(f"{prefix}%"))
                .group_by(models.StudyModel.study_name)
            )
            
            results = query.all()
            
            # Build dictionary mapping study names to completed trial counts
            counts_by_study = {study_name: count for study_name, count in results}
            return counts_by_study
    
    def _get_cached_trial_count(self, study_name: str, state: str, study_name_prefix: Optional[str] = None) -> Optional[int]:
        """
        Get cached trial count for a specific study and state.
        
        Args:
            study_name (str): The study name
            state (str): The trial state to count
            study_name_prefix (Optional[str]): Specific prefix cache to check
        
        Returns:
            Optional[int]: The count if found in cache, None otherwise
        """
        if state != optuna.trial.TrialState.COMPLETE:
            return None
            
        if study_name_prefix:
            cache = self.prefix_caches.get(study_name_prefix)
            if cache and cache.loaded and study_name in cache.completed_trial_counts_by_study:
                return cache.completed_trial_counts_by_study.get(study_name, 0)
        else:
            for cache in self.prefix_caches.values():
                if cache.loaded and study_name in cache.completed_trial_counts_by_study:
                    return cache.completed_trial_counts_by_study.get(study_name, 0)
        
        return None
    
    def count_trial_state(self, study_name: str, state: str, study_name_prefix: Optional[str] = None, allow_cache: bool = True) -> int:
        """
        Count trials of a specific state for a study.
        
        Args:
            study_name (str): The study name
            state (str): The trial state to count
            study_name_prefix (Optional[str]): Specific prefix cache to check
            allow_cache (bool): Whether to use cached counts
        
        Returns:
            int: Count of trials with the specified state
        """
        # Check if we have cached completed counts
        if allow_cache:
            cached_count = self._get_cached_trial_count(study_name, state, study_name_prefix)
            if cached_count is not None:
                return cached_count
        
        # Query the database
        with self._create_session() as session:
            query = session.query(
                sqlalchemy.func.count(models.TrialModel.trial_id)
            ).join(
                models.StudyModel, 
                models.StudyModel.study_id == models.TrialModel.study_id
            ).filter(
                models.StudyModel.study_name == study_name,
                models.TrialModel.state == state
            )
            
            result = query.scalar()
            return result or 0
    
    def study_exists(self, study_name: str, prefix: Optional[str] = None) -> bool:
        """Check if a study with the given name exists.
        
        Args:
            study_name: The name of the study to check
            prefix: Optional prefix to use for caching checks
            
        Returns:
            bool: True if the study exists, False otherwise
        """
        # If we have a prefix and it's in our cache, check there first
        if prefix and prefix in self.prefix_caches:
            cache = self.prefix_caches[prefix]
            if cache.loaded:
                return study_name in cache.completed_trial_counts_by_study
        
        # If not found in cache or no prefix provided, check directly in storage
        try:
            # Use a direct query to check if the study exists
            with self._create_session() as session:
                study_id = self._engine.get_study_id_from_name(session, study_name)
                return study_id is not None
        except Exception as e:
            logger.error(f"Error checking if study '{study_name}' exists: {e}")
            return False
    
    def populate_study_trial_caches(self, studies: Iterable[optuna.Study]) -> None:
        """
        Efficiently caches trials for multiple studies in a single database operation.
        
        This method directly populates each study's internal cache (_thread_local.cached_all_trials)
        to avoid repetitive database calls when accessing trials.
        
        Args:
            studies: An iterable of optuna.Study objects to cache trials for
            study_name_prefix: Prefix to filter studies by name instead of using an IN clause
                               with study IDs. This is more efficient for large numbers of studies.
        """
        # logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)  # Use DEBUG for even more detail
        logger.info(f"Caching trials for {len(studies)} studies.")
        study_by_id = {study._study_id: study for study in studies}
        populate_start_time = time.perf_counter()
        if not study_by_id:
            return
        
        try:
            # Fetch all trials for all requested studies in a single query
            with self._create_session() as session:
                query = (
                    session.query(models.TrialModel)
                    .options(sqlalchemy_orm.subqueryload(models.TrialModel.params))
                    .options(sqlalchemy_orm.subqueryload(models.TrialModel.values))
                    .options(sqlalchemy_orm.subqueryload(models.TrialModel.user_attributes))
                    .options(sqlalchemy_orm.subqueryload(models.TrialModel.system_attributes))
                    .options(sqlalchemy_orm.subqueryload(models.TrialModel.intermediate_values))
                )
                query = (
                    query.join(
                        models.StudyModel,
                        models.StudyModel.study_id == models.TrialModel.study_id
                    )
                    .filter(models.StudyModel.study_id.in_(study_by_id.keys()))
                ).order_by(models.TrialModel.study_id)
                
                trial_models = query.all()
                trials_by_study_id = {study_id: [self.storage._backend._build_frozen_trial_from_trial_model(trial_model) 
                                                    for trial_model in group] 
                                        for study_id, group in itertools.groupby(trial_models, lambda x: x.study_id)}
                
                # Populate each study's internal cache
                for study_id, trials in trials_by_study_id.items():
                    study = study_by_id[study_id]
                    # Replace the get_trials method to use the eager version. use default argument to bind the correct trials variable here.
                    study.get_trials = lambda trials = trials, **_: trials
            logger.info(f"Loaded {len(trial_models)} trials in {perf_log_text(populate_start_time)}")
        except Exception as e:
            logger.exception(f"Error loading trials for studies: {e}")
            raise e
    