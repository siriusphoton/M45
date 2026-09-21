from m45.agent import create_application_agent
from m45.config import load_settings

graph = create_application_agent(load_settings())
