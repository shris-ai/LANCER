from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
import yaml
import json
from src.logger_setup import logger
from src.embedding_db import VectorDB


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class LLM_Agent:
    def __init__(self, model: str = "gpt-4o", config_path: str=None):
        """
        Initializes the OpenAI client with the selected model.

        Args:
            model_choice (str): Either "gpt-4o" or "o3-mini".
        """
        self.model = model
        self.config = self.load_yaml_prompt(config_path)
        self.name = self.config.get("name")
        self.role = self.config.get("role")
        self.description = self.config.get("description")
        self.goals = self.config.get("goals", [])
        self.inputs = self.config.get("inputs", [])
        self.outputs = self.config.get("outputs", {})
        self.notes = self.config.get("notes", "")
        self.guidelines = self.config.get("guidelines", "")

    def load_yaml_prompt(self, filepath):
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
        
    def prompt_template(self):
        return f"""You are {self.role}.

        {self.description}

        Goals:
        {chr(10).join(f"- {g}" for g in self.goals)}
        
        Inputs:
        {', '.join(self.inputs)}
        
        Output Format: {self.outputs.get('format')}
        
        Schema:
        {self.outputs.get('schema')}
        
        Notes:
        {self.notes}
        
        Guidelines:
        {self.guidelines}
        """

    def get_response(self, messages) -> str:
        """
        Sends a prompt to the OpenAI model and returns the response.

        Args:
            prompt (str): The input prompt.

        Returns:
            str: The model's response.
        """
        completion = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return completion.choices[0].message.content

class Reasoning_Agent(LLM_Agent):
    def __init__(self, model: str = "o3-mini"):
        """
        Initializes the OpenAI client with the selected model.

        Args:
            model_choice (str): Either "gpt-4o" or "o3-mini".
        """
        self.model = model

class PlanningAgent(LLM_Agent):
    def __init__(self, model: str = "o3-mini"):
        super().__init__(model, "src/prompts/planning.yaml")
        
    def plan_steps(self, problem_description: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.prompt_template()
            },
            {
                "role": "user",
                "content": f"Problem Description:\n{problem_description}"
            }
        ]
        return self.get_response(messages)

class CodeGenAgent(LLM_Agent):
    def __init__(self, model: str = "gpt-4o"):
        super().__init__(model, "src/prompts/generation.yaml")

    def generate_code_and_proof(self, plan: str, template: str, retrieved_context: str = "") -> str:
        messages = [
            {
                "role": "system",
                "content": self.prompt_template()
            },
            {
                "role": "user",
                "content": (
                    f"Plan:\n{plan}\n\n"
                    f"Template:\n{template}\n\n"
                    f"{'Retrieved Context:\n' + retrieved_context if retrieved_context else ''}"
                )
            }
        ]
        return self.get_response(messages)

class VerificationAgent(LLM_Agent):
    def __init__(self, model: str = "o3-mini",  execute_lean_code_func=None):
        super().__init__(model, "src/prompts/verification.yaml")

        # Save the function reference for execution
        self.execute_lean_code = execute_lean_code_func

    def verify_code_and_proof(self, problem: str, template: str, code: str, proof: str, lean_output_stdout: str, lean_output_stderr: str) -> str:
        messages = [
            {
                "role": "system",
                "content": self.prompt_template()
            },
            {
                "role": "user",
                "content": (
                    f"Problem Description:\n{problem}\n\n"
                    f"Lean Template:\n{template}\n\n"
                    f"Generated Implementation:\n{code}\n\n"
                    f"Generated Proof:\n{proof}\n\n"
                    f"Lean Output STDOUT:\n{lean_output_stdout}\n\n"
                    f"Lean Output STDERR:\n{lean_output_stderr}"
                )
            }
        ]
        return self.get_response(messages)

    def run_and_verify(self, problem: str, template: str, code: str, proof: str) -> str:
        if self.execute_lean_code is None:
            raise ValueError("No execute_lean_code function provided to VerificationAgent.")
        # Substitute code and proof into the template
        lean_source = template.replace("{{code}}", code).replace("{{proof}}", proof)

        result = self.execute_lean_code(lean_source)
        logger.info(f"Lean execution result: {result}")

        lean_output_stdout = ""
        lean_output_stderr = ""

        if isinstance(result, dict):
            lean_output_stdout = result.get("stdout", "")
            lean_output_stderr = result.get("stderr", "")
        elif isinstance(result, (list, tuple)):
            # Assume first two elements are stdout and stderr
            lean_output_stdout = result[0] if len(result) > 0 else ""
            lean_output_stderr = result[1] if len(result) > 1 else ""
        elif isinstance(result, str):
            # If only a string is returned, treat it as stderr
            lean_output_stderr = result
        else:
            lean_output_stderr = str(result)

        return self.verify_code_and_proof(
            problem=problem,
            template=template,
            code=code,
            proof=proof,
            lean_output_stdout=lean_output_stdout,
            lean_output_stderr=lean_output_stderr
        )
        
class AgentWorkflow:
    def __init__(
        self,
        planning_agent: PlanningAgent,
        codegen_agent: CodeGenAgent,
        verification_agent: VerificationAgent,
        max_attempts: int = 3
    ):
        self.planning_agent = planning_agent
        self.codegen_agent = codegen_agent
        self.verification_agent = verification_agent
        self.max_attempts = max_attempts

    def run(self, problem_description: str, task_lean_code: str) -> dict:
        top_k_chunks,_ = VectorDB.run(problem_description)
        rag_context = "\n".join(top_k_chunks)
        logger.info(f"Retrieved context: {rag_context}")
        print(f"Retrieved context: {rag_context}")
        plan = self.planning_agent.plan_steps(problem_description)
        previous_errors = set()

        for attempt in range(self.max_attempts):
            logger.info(f"Attempt {attempt + 1}/{self.max_attempts}")
            logger.info(f"Current plan: {plan}")
            if attempt > 0 and feedback:
                # Extract retry guidance and error summary from feedback
                retry_guidance = feedback.get("retry_strategy", "")
                error_summary = feedback.get("error_summary", "")
                # Append to plan for codegen agent
                plan += (
                    f"\n\nLean error summary:\n{error_summary}\n"
                    f"Retry guidance (MUST FOLLOW):\n{retry_guidance}\n"
                    "You MUST NOT use 'sorry' in code or proof. "
                    "If you do, your output will be rejected. "
                    "Always provide a complete implementation and proof."
                    )
                logger.info(f"Updated plan: {plan}")

            
            # Generate code and proof
            raw_solution = self.codegen_agent.generate_code_and_proof(
                plan, task_lean_code, retrieved_context=rag_context
            )
            logger.info(f"Raw solution received: {raw_solution}")
            logger.info(f"{type(raw_solution)}")

            if raw_solution.strip().startswith("```"): 
                # Remove the first line (```json or ```) and the last line (```)
                lines = raw_solution.strip().splitlines()
                # Remove the first and last line if they are code block markers
                if lines[0].startswith("```") and lines[-1].startswith("```"):
                    raw_solution = "\n".join(lines[1:-1])
            
            try:
                generated_solution = json.loads(raw_solution)
                logger.info(f"Generated solution: {generated_solution}")
            except Exception:
                generated_solution = {"code": "sorry", "proof": "sorry"}
                logger.error("Failed to parse generated solution as JSON. Using default 'sorry' values.")
                logger.error(f"{Exception}")
                logger.error(f"Fallback Generated solution: {generated_solution}")

            code = generated_solution.get("code", "sorry")
            proof = generated_solution.get("proof", "sorry")

            # Verify code and proof
            raw_feedback = self.verification_agent.run_and_verify(
                problem_description, task_lean_code, code, proof
            )
            try:
                feedback = json.loads(raw_feedback)
                logger.info(f"Feedback received: {feedback}")
            except Exception:
                feedback = {"verdict": "fail", "error_summary": raw_feedback, "error_type": "unknown"}

            # If verification passes, return solution
            if feedback.get("verdict") == "pass":
                return {"code": code, "proof": proof}

            # If error is repeated, break to avoid infinite loop
            error_message = feedback.get("error_summary", "")
            if error_message in previous_errors:
                logger.info(f"Attempt {attempt + 1} failed due to repeated error: {error_message}")
                logger.error("Last attempt failed. Returning original code and proof.")
                return {"code": code, "proof": proof}
            previous_errors.add(error_message)

            if feedback.get("retrieval_prompt") is None:
                retrieval_prompt = feedback.get("retrieval_prompt", "")
                top_k_chunks,_ = VectorDB.run(retrieval_prompt)
                rag_context = "\n".join(top_k_chunks)
                logger.info(f"Updated Retrieved context: {rag_context}")
                print(f"Updated Retrieved context: {rag_context}")


            # Feedback loop: update plan based on error
            plan = self.planning_agent.plan_steps(
                f"{problem_description}\n\nPrevious plan:\n{plan}\n\nLean error:\n{error_message}\n\n"
                "Revise your plan to address the above Lean error."
            )

        # If all attempts fail, return sorry
        return {"code": code, "proof": proof}

# Example usage:
if __name__ == "__main__":
    agent = LLM_Agent(model="gpt-4o")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain the concept of recursion in programming?"}
    ]
    #response = agent.get_response(messages)
    #print(response)
    
    reasoning_agent = Reasoning_Agent(model="o3-mini")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Can you explain the concept of recursion in programming?"}
    ]
    solution = reasoning_agent.get_response(messages)
    print(solution)