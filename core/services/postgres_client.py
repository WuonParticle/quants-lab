import logging
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from typing import Dict, Tuple, Any, Optional, Union

logger = logging.getLogger(__name__)

class PostgresClient:
    """
    A client for PostgreSQL database operations, especially for creating 
    databases if they don't exist.
    """
    
    def __init__(self, host: str = "localhost", port: int = 5432, 
                 user: str = "admin", password: str = "admin", database: str = "postgres"):
        """
        Initialize the PostgreSQL client.
        
        Args:
            host (str): Database host
            port (int): Database port
            user (str): Database username
            password (str): Database password
            database (str): Default database name (usually 'postgres' for admin operations)
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
    
    def ensure_database_exists(self, db_name: str) -> bool:
        """
        Check if the specified database exists and create it if it doesn't.
        
        Args:
            db_name (str): The name of the database to check/create
            
        Returns:
            bool: True if the database exists or was created successfully, False otherwise
        """
        try:
            # Connect to default database to check if our target db exists
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                dbname=self.database
            )
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
            exists = cursor.fetchone()
            
            if not exists:
                logger.info(f"Database '{db_name}' does not exist. Creating it...")
                cursor.execute(f"CREATE DATABASE {db_name}")
                logger.info(f"Database '{db_name}' created successfully.")
            else:
                logger.debug(f"Database '{db_name}' already exists.")
                
            cursor.close()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring database exists: {str(e)}")
            return False
    
    @staticmethod
    def from_config(**kwargs) -> Tuple['PostgresClient', str]:
        """
        Create a PostgresClient from configuration options and generate a SQLAlchemy connection string.
        
        This method handles both postgres_config dictionary and individual parameters.
        
        Args:
            config (Dict): A dictionary containing postgres configuration
                          with keys: host, port, user, password, database
            **kwargs: Alternative way to pass parameters, supports:
                      db_host, db_port, db_user, db_password, database_name
                      create_db_if_not_exists (bool): Whether to create the database if it doesn't exist
        
        Returns:
            Tuple[PostgresClient, str]: A tuple containing (postgresql_client, connection_string)
        """
        # Support "db_host" for backwards compatibility
        if "db_host" in kwargs: 
            db_host = kwargs.get("db_host", "localhost")
            db_port = kwargs.get("db_port", 5432)
            db_user = kwargs.get("db_user", "admin")
            db_pass = kwargs.get("db_password", "admin")
            db_name = kwargs.get("database_name", "optimization_database")
        elif "postgres_config" in kwargs:
            db_config = kwargs["postgres_config"]
            db_host = db_config.get("host", "localhost")
            db_port = db_config.get("port", 5432)
            db_user = db_config.get("user", "admin")
            db_pass = db_config.get("password", "admin")
            db_name = db_config.get("database", "optimization_database")
        
        # Create the connection string for SQLAlchemy
        connection_string = f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        
        # Create client (connects to postgres database, not the target database)
        postgres_client = PostgresClient(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_pass,
            database="postgres"  # Connect to default postgres db to create new db
        )
        
        # Ensure the database exists if requested
        create_db = kwargs.get("create_db_if_not_exists", True)
        if create_db:
            success = postgres_client.ensure_database_exists(db_name)
            if not success:
                logger.error(f"Failed to create database '{db_name}'.")
        
        return postgres_client, connection_string
    
    @classmethod
    def from_connection_string(cls, connection_string: str):
        """
        Create a PostgresClient from a connection string.
        
        Args:
            connection_string (str): A connection string in the format:
                                    postgresql+psycopg2://user:pass@host:port/dbname
        
        Returns:
            PostgresClient: A configured PostgresClient instance
        """
        # Parse connection string to extract components
        conn_parts = connection_string.replace("postgresql+psycopg2://", "").split("/")
        conn_string = conn_parts[0]
        db_name = conn_parts[1] if len(conn_parts) > 1 else "postgres"
        
        # Extract credentials and connection info
        credentials, host_port = conn_string.split("@")
        user_pass = credentials.split(":")
        username = user_pass[0]
        password = user_pass[1] if len(user_pass) > 1 else ""
        
        host_port_parts = host_port.split(":")
        host = host_port_parts[0]
        port = int(host_port_parts[1]) if len(host_port_parts) > 1 else 5432
        
        return cls(host=host, port=port, user=username, password=password)