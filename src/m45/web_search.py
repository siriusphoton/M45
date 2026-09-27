from langchain_core.tools import BaseTool, tool
from ollama import Client, WebSearchResponse

WEB_SEARCH_TOOL_NAME = "web_search"
WEB_SEARCH_RESULT_LIMIT = 3


def create_ollama_web_search_tool(api_key: str) -> BaseTool:
    @tool(WEB_SEARCH_TOOL_NAME)
    def web_search(query: str) -> str:
        """Search the public web for current or externally verifiable information.

        Keep the query concise. Do not include private personal details unless the
        user explicitly requests a personalized search and those details are needed.
        """
        client = Client(headers={"Authorization": f"Bearer {api_key}"})
        try:
            response: WebSearchResponse = client.web_search(
                query=query,
                max_results=WEB_SEARCH_RESULT_LIMIT,
            )
        finally:
            client.close()

        if not response.results:
            return "No web search results found."

        return "\n\n".join(
            (
                f"Title: {result.title or 'Untitled result'}\n"
                f"URL: {result.url or 'URL unavailable'}\n"
                f"Snippet: {result.content or 'No snippet available.'}"
            )
            for result in response.results
        )

    return web_search
