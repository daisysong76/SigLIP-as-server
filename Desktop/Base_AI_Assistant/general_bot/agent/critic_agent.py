# agent/critic_agent.py
class CriticAgent:
    def evaluate(self, results: list) -> str:
        """
        Evaluate the results of the executed tasks and return an evaluation report.
        """
        evaluation = "Evaluation Report:\n"
        for result in results:
            evaluation += f"- {result}\n"
        return evaluation
