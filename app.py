# from agno.agent import Agent
# from agno.models.ollama import Ollama
# from dotenv import load_dotenv
# from agno.db.sqlite import SqliteDb
# from agno.tools.csv_toolkit import CsvTools
# from agno.tools.file import FileTools
# from agno.tools.pandas import PandasTools
# from agno.tools.visualization import VisualizationTools
# from agno.team import Team
# from agno.os import AgentOS
# from pathlib import Path

# # ================= ENV =================
# load_dotenv()

# BASE_DIR = Path(__file__).parent
# DATA_DIR = BASE_DIR / "data"

# csv_files = list(DATA_DIR.glob("*.csv"))
# if not csv_files:
#     raise FileNotFoundError("No CSV files found in data/ folder")

# # ================= DB =================
# db = SqliteDb(
#     db_file="memory.db",
#     session_table="session_table"
# )

# # ================= MODEL =================
# model = Ollama("qwen2.5:3b-instruct")

# # ================= DATA LOADER AGENT =================
# data_loader_agent = Agent(
#     id="data-loader-agent",
#     name="Data Loader Agent",
#     model=model,
#     db=db,
#     role="CSV discovery only",
#     instructions=[
#         "You ONLY confirm CSV existence.",
#         "CSV files are ONLY inside the data folder.",
#         "Do NOT perform pandas operations."
#     ],
#     tools=[
#         CsvTools(csvs=csv_files),
#         FileTools(base_dir=BASE_DIR)
#     ]
# )

# # ================= DATA UNDERSTANDING AGENT =================
# data_understanding_agent = Agent(
#     id="data-understanding-agent",
#     name="Data Understanding Agent",
#     model=model,
#     db=db,
#     role="All dataframe operations",
#     add_history_to_context=True,
#     instructions=[
#         "You are a dataframe EXECUTOR, not a planner.",
#         "When a task is delegated to you, you MUST immediately run tools.",
#         "NEVER describe steps without executing them.",
#         "",
#         "MANDATORY EXECUTION ORDER:",
#         "1. Create dataframe from CSV using PandasTools.create_dataframe_from_csv.",
#         "   - Path MUST be exactly: data/car_details.csv",
#         "   - Dataframe name MUST be exactly: car_df",
#         "",
#         "2. After dataframe creation, run ALL requested operations on car_df.",
#         "",
#         "RULES:",
#         "- NEVER invent dataframe names.",
#         "- NEVER skip dataframe creation.",
#         "- ALWAYS pass operation_parameters as {}.",
#         "- Continue running tools until the task is fully completed.",
#         "- Return the FINAL results (tables + brief explanation)."
#     ],
#     tools=[
#         PandasTools(),
#         CsvTools(csvs=csv_files),
#         FileTools(base_dir=BASE_DIR)
#     ]
# )

# # ================= VISUALIZATION AGENT =================
# visualization_agent = Agent(
#     id="viz-agent",
#     name="Visualization Agent",
#     model=model,
#     db=db,
#     role="Plot creation only",
#     instructions=[
#         "Create plots ONLY after dataframe car_df exists.",
#         "Use matplotlib via VisualizationTools."
#     ],
#     tools=[
#         PandasTools(),
#         VisualizationTools(output_dir="plots"),
#         FileTools(base_dir=BASE_DIR)
#     ]
# )

# # ================= TEAM =================
# data_science_team = Team(
#     id="data-science-team",
#     name="Data Science Assistant",
#     model=model,
#     members=[
#         data_loader_agent,
#         data_understanding_agent,
#         visualization_agent
#     ],
#     role="Team Leader / Orchestrator",
#     instructions=[
#         "You are the orchestrator.",
#         "",
#         "CRITICAL RULE:",
#         "Delegation is NOT completion.",
#         "",
#         "When you delegate a task to a member:",
#         "- You MUST wait for the member to EXECUTE tools.",
#         "- You MUST collect the member’s FINAL output.",
#         "- You MUST return the results to the user.",
#         "",
#         "ROUTING RULES:",
#         "- Any request involving rows, columns, statistics, tables → data-understanding-agent.",
#         "- Any request involving plots → visualization-agent.",
#         "",
#         "You are NOT allowed to stop after delegation.",
#         "You must always return executed results."
#     ],
#     db=db,
#     add_history_to_context=True,
#     add_member_tools_to_context=True,
#     enable_agentic_state=True,
#     num_history_runs=5
# )

# # ================= AGENT OS =================
# agent_os = AgentOS(
#     id="agent-os",
#     name="data-science-assistant",
#     description="Execution-safe multi-agent data science assistant",
#     teams=[data_science_team]
# )

# # ================= FASTAPI APP =================
# app = agent_os.get_app()

# # ================= MAIN =================
# if __name__ == "__main__":
#     agent_os.serve(app="app:app")
