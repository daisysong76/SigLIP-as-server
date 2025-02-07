import asyncio

class PlanAgent:
    async def generate_plan(self, instructions: dict) -> list:
        """
        Generate a plan (a list of tasks) based on the provided instructions.
        Each task includes its title, last commit message, and last commit date.
        """
        await asyncio.sleep(0.5)  # Simulate planning delay
        
        # Mapping each task to its details based on your table.
        task_definitions = {
            "browsing": {
                "title": "Browsing",
                "last_commit_message": "initialize project with turborepo",
                "last_commit_date": "2 days ago"
            },
            "calendar": {
                "title": "Calendar",
                "last_commit_message": "initialize project with turborepo",
                "last_commit_date": "2 days ago"
            },
            "call": {
                "title": "Call",
                "last_commit_message": "add onto main",
                "last_commit_date": "2 days ago"
            },
            "contact_search": {
                "title": "Contact Search",
                "last_commit_message": "initialize project with turborepo",
                "last_commit_date": "2 days ago"
            },
            "email": {
                "title": "Email",
                "last_commit_message": "initialize project with turborepo",
                "last_commit_date": "2 days ago"
            },
            "search": {
                "title": "Search",
                "last_commit_message": "initialize project with turborepo",
                "last_commit_date": "2 days ago"
            },
            # New device-control tasks:
            "keyboard": {
                "title": "Keyboard Input",
                "last_commit_message": "simulate keyboard input",
                "last_commit_date": "today"
            },
            "mouse": {
                "title": "Mouse Control",
                "last_commit_message": "simulate mouse click",
                "last_commit_date": "today"
            },
            "vision": {
                "title": "Screen Capture",
                "last_commit_message": "capture screenshot",
                "last_commit_date": "today"
            },
            "webbrowse": {
                "title": "Web Browser Control",
                "last_commit_message": "simulate browser action",
                "last_commit_date": "today"
            }
        }

        plan = []
        for task_key in instructions.get("requested_tasks", []):
            details = task_definitions.get(task_key)
            if details:
                plan.append({
                    "task": task_key,
                    "details": details
                })
            else:
                # If the task is unknown, use default values.
                plan.append({
                    "task": task_key,
                    "details": {
                        "title": task_key.title(),
                        "last_commit_message": "N/A",
                        "last_commit_date": "N/A"
                    }
                })
        return plan


