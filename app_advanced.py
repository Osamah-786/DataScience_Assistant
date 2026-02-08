"""
Advanced Multi-Agent Data Science System
Optimized for Agno Platform Deployment
"""

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.db.sqlite import SqliteDb
from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from agno.tools.visualization import VisualizationTools
from agno.team import Team
from agno.os import AgentOS

from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging
import json


# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("agent_system.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ================= ENV =================
load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / "cache"

for d in [DATA_DIR, PLOTS_DIR, REPORTS_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)


# ================= CONFIG =================
class Config:
    MODEL_NAME = "qwen2.5:7b-instruct"
    DB_FILE = "memory.db"
    SESSION_TABLE = "session_table"
    CONTEXT_HISTORY_RUNS = 10
    MAX_TOOL_CALLS = 50


# ================= DATA VALIDATION =================
class DataValidator:
    @staticmethod
    def validate_csv_files(data_dir: Path) -> List[Path]:
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        logger.info(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")
        return csv_files

    @staticmethod
    def get_file_metadata(csv_files: List[Path]) -> Dict[str, Any]:
        return {
            f.name: {
                "size_mb": f.stat().st_size / (1024 * 1024),
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            }
            for f in csv_files
        }


# ================= INIT =================
logger.info("Initializing Advanced Data Science Agent System")

csv_files = DataValidator.validate_csv_files(DATA_DIR)
file_metadata = DataValidator.get_file_metadata(csv_files)
logger.info(f"File metadata: {json.dumps(file_metadata, indent=2)}")


# ================= DB =================
db = SqliteDb(db_file=Config.DB_FILE, session_table=Config.SESSION_TABLE)
logger.info("Database initialized")


# ================= MODEL =================
model = Ollama(Config.MODEL_NAME)
logger.info(f"Model initialized: {Config.MODEL_NAME}")


# ================= AGENTS =================
data_discovery_agent = Agent(
    id="data-discovery-agent",
    name="Data Discovery Agent",
    model=model,
    db=db,
    role="Data discovery",
    instructions=[
        "Identify CSV files in data/.",
        "Provide metadata and quality suggestions.",
        "Do NOT perform pandas analysis.",
    ],
    tools=[CsvTools(csvs=csv_files), FileTools(base_dir=BASE_DIR)],
    add_history_to_context=True,
)

data_analysis_agent = Agent(
    id="data-analysis-agent",
    name="Data Analysis Agent",
    model=model,
    db=db,
    role="Advanced analysis",
    instructions=[
        "Create dataframe using PandasTools.create_dataframe_from_csv.",
        "Perform statistics, cleaning, correlations, anomalies.",
        "Return structured insights.",
    ],
    tools=[PandasTools(), CsvTools(csvs=csv_files), FileTools(base_dir=BASE_DIR)],
    add_history_to_context=True,
    num_history_runs=Config.CONTEXT_HISTORY_RUNS,
)

visualization_agent = Agent(
    id="visualization-agent",
    name="Visualization Agent",
    model=model,
    db=db,
    role="Plotting",
    instructions=[
        "Generate charts ONLY when explicitly requested.",
        "Save plots to the plots/ directory.",
    ],
    tools=[
        PandasTools(),
        VisualizationTools(output_dir=str(PLOTS_DIR)),
        FileTools(base_dir=BASE_DIR),
    ],
    add_history_to_context=True,
)

statistical_agent = Agent(
    id="statistical-insights-agent",
    name="Statistical Insights Agent",
    model=model,
    db=db,
    role="Statistical reasoning",
    instructions=[
        "Provide descriptive and inferential statistics with interpretation."
    ],
    tools=[PandasTools(), FileTools(base_dir=BASE_DIR)],
    add_history_to_context=True,
)

report_agent = Agent(
    id="report-generation-agent",
    name="Report Generation Agent",
    model=model,
    db=db,
    role="Final report synthesis",
    instructions=[
        "Generate the final professional report.",
        "Reference plot filenames if visualizations exist.",
        "Return only the final answer.",
    ],
    tools=[FileTools(base_dir=BASE_DIR), PandasTools()],
    add_history_to_context=True,
)


# ================= TEAM =================
data_science_team = Team(
    id="advanced-data-science-team",
    name="Advanced Data Science Team",
    model=model,
    members=[
        data_discovery_agent,
        data_analysis_agent,
        visualization_agent,
        statistical_agent,
        report_agent,
    ],
    role="Lead orchestrator",
    instructions=[
        "You are a strict workflow orchestrator.",

        "CRITICAL RULE:",
        "You MUST delegate tasks using the EXACT agent IDs:",
        "data-discovery-agent, data-analysis-agent, visualization-agent,",
        "statistical-insights-agent, report-generation-agent.",
        "NEVER refer to agents by numbers or descriptions.",

        "EXECUTION ORDER:",
        "1. data-discovery-agent",
        "2. data-analysis-agent",
        "3. IF plots requested → visualization-agent",
        "4. statistical-insights-agent",
        "5. report-generation-agent",

        "Stop reasoning loops. Always finish with final report.",
    ],
    db=db,
    add_history_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True,
    num_history_runs=Config.CONTEXT_HISTORY_RUNS,
)


# ================= AGENT OS =================
agent_os = AgentOS(
    id="advanced-agent-os",
    name="Advanced Data Science Assistant",
    description="Enterprise multi-agent data science system",
    teams=[data_science_team],
)


# ================= ASGI =================
app = agent_os.get_app()


# ================= MAIN =================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Advanced Data Science Agent System Ready")
    logger.info("=" * 60)
    logger.info(f"CSV Files: {[f.name for f in csv_files]}")
    logger.info("=" * 60)

    agent_os.serve(app="app_advanced3:app")
