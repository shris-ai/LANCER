import os
import json
import argparse
from src.agents import Reasoning_Agent, LLM_Agent, PlanningAgent, CodeGenAgent, VerificationAgent, AgentWorkflow
from src.embedding_db import VectorDB
from src.lean_runner import execute_lean_code
from typing import Dict, List, Tuple
from src.logger_setup import logger



type LeanCode = Dict[str, str]

def main_workflow(problem_description: str, task_lean_code: str = "") -> LeanCode:
    """
    Main workflow for the coding agent. This workflow takes in the problem description in natural language (description.txt) 
    and the corresponding Lean code template (task.lean). It returns the function implementation and the proof in Lean.
    
    Args:
        problem_description: Problem description in natural language. This file is read from "description.txt"
        task_lean_code: Lean code template. This file is read from "task.lean"
    
    Returns:
        LeanCode: Final generated solution, which is a dictionary with two keys: "code" and "proof".
    """
    generated_function_implementation = "sorry"
    generated_proof = "sorry"

    # TODO Implement your coding workflow here. The unit tests will call this function as the main workflow.
    # Feel free to chain multiple agents together, use the RAG database (.pkl) file, corrective feedback, etc.
    # Please use the agents provided in the src/agents.py module, which include GPT-4o and the O3-mini models.
    ...
    print(f"Problem description: {problem_description}")
    print(f"Task Lean code: {task_lean_code}")

    logger.info(f"Problem description: {problem_description}")
    logger.info(f"Task Lean code: {task_lean_code}")

    '''
    top_k_chunks,_ = VectorDB.run(problem_description)
    context = "\n".join(top_k_chunks)
    logger.info(f"Retrieved context: {context}")
    print(f"Retrieved context: {context}")
    '''

    '''
    planning_agent = PlanningAgent()
    plan = planning_agent.plan_steps(problem_description)
    logger.info(f"Generated plan: {plan}")
    print(f"Generated plan: {plan}")

    codegen_agent = CodeGenAgent()
    raw_solution = codegen_agent.generate_code_and_proof(plan, task_lean_code, retrieved_context=context)
    generated_solution = json.loads(raw_solution)
    logger.info(f"Generated solution: {generated_solution}")
    print(f"Generated solution: {generated_solution}")
    generated_function_implementation = generated_solution["code"]
    print(f"Generated code: {generated_function_implementation}")
    logger.info(f"Generated code: {generated_function_implementation}")
    generated_proof = generated_solution["proof"]
    logger.info(f"Generated proof: {generated_proof}")
    print(f"Generated proof: {generated_proof}")

    verification_agent = VerificationAgent(execute_lean_code_func=execute_lean_code)

    raw_feedback = verification_agent.run_and_verify(
        problem_description,
        task_lean_code,
        generated_function_implementation,
        generated_proof
    )

    verification_result = json.loads(raw_feedback)
    logger.info(f"Verification result: {verification_result}")
    print(f"Verification result: {verification_result}")
    if verification_result["verdict"] == "pass":
        logger.info("Verification passed.")
        print("Verification passed.")
        generated_function_implementation = generated_function_implementation
        generated_proof = generated_proof
    else:
        logger.error(f"Verification failed: {verification_result['error_summary']}")
        print(f"Verification failed: {verification_result['error_summary']}")
        # Handle the case where verification fails, e.g., by returning an error or retrying.
        return {
            "code": "sorry",
            "proof": "sorry"
        }
    '''
    # Instantiate agents
    planning_agent = PlanningAgent()
    codegen_agent = CodeGenAgent()
    verification_agent = VerificationAgent(execute_lean_code_func=execute_lean_code)

    # Use the feedback loop workflow
    workflow = AgentWorkflow(planning_agent, codegen_agent, verification_agent)
    result = workflow.run(problem_description, task_lean_code)

    # Unpack for required return
    generated_function_implementation = result.get("code", "sorry")
    generated_proof = result.get("proof", "sorry")

    # Example return for task_id_0
    #generated_function_implementation = "x"
    #generated_proof = "rfl"
    
    return {
        "code": generated_function_implementation,
        "proof": generated_proof
    }

def get_problem_and_code_from_taskpath(task_path: str) -> Tuple[str, str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and also read the file 
    that contains the task description, which is "description.txt".
    
    After reading the files, it will return a tuple of the problem description and the Lean code template.
    
    Args:
        task_path: Path to the task file
    """
    problem_description = ""
    lean_code_template = ""
    
    with open(os.path.join(task_path, "description.txt"), "r") as f:
        problem_description = f.read()

    with open(os.path.join(task_path, "task.lean"), "r") as f:
        lean_code_template = f.read()

    return problem_description, lean_code_template

def get_unit_tests_from_taskpath(task_path: str) -> List[str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "tests.lean" and return the unit tests.
    """
    with open(os.path.join(task_path, "tests.lean"), "r") as f:
        unit_tests = f.read()
    
    return unit_tests

def get_task_lean_template_from_taskpath(task_path: str) -> str:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and return the Lean code template.
    """
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        task_lean_template = f.read()
    return task_lean_template