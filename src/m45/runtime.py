from m45.agent import create_application_agent
from m45.config import load_settings
from m45.database import create_database_engine

settings = load_settings()
database_engine = create_database_engine(settings)

graph = create_application_agent(settings, database_engine)
