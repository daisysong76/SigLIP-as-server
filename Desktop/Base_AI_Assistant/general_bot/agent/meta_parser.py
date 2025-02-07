
# agent/meta_parser.py
def parse_meta_input(meta_input: str) -> dict:
    """
    Parse the raw meta input into a structured instruction.
    If the input is "all tasks", return a dictionary with a list of pre-defined tasks.
    Otherwise, assume a comma-separated list of tasks.
    """
    if meta_input.strip().lower() == "all tasks":
        return {
            "requested_tasks": [
                "browsing",
                "calendar",
                "call",
                "contact_search",
                "email",
                "search"
            ]
        }
    else:
        # Assume comma-separated list of tasks (normalize to lowercase with underscores)
        tasks = [task.strip().lower().replace(" ", "_") for task in meta_input.split(",")]
        return {"requested_tasks": tasks}

