"""
Multi-Model Advanced Data Science System for Agno Platform
Production-Ready Version with Automatic Workflow

Models:
- llama3.1:8b → Team Leader (orchestration)
- qwen2.5:14b-instruct → Data Analysis (complex computations)
- mistral:7b-instruct → Statistical Analysis (mathematical reasoning)
- qwen2.5:7b-instruct → Visualization (chart generation)
- llama3.2:3b → Data Discovery (simple tasks)
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

from pathlib import Path
from datetime import datetime
import logging
import json

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================= DIRECTORY SETUP =================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / "cache"

# Create all directories automatically
for d in [DATA_DIR, PLOTS_DIR, REPORTS_DIR, CACHE_DIR]:
    d.mkdir(exist_ok=True)
    logger.info(f"✓ Directory ready: {d}")

# ================= CONFIG =================
class ModelConfig:
    """Multi-model configuration for specialized tasks"""
    
    LEADER_MODEL = "llama3.1:8b"
    ANALYSIS_MODEL = "qwen2.5:14b-instruct"
    STATS_MODEL = "mistral:7b-instruct"
    VIZ_MODEL = "qwen2.5:7b-instruct"
    DISCOVERY_MODEL = "llama3.2:3b"
    
    DB_FILE = "memory.db"
    SESSION_TABLE = "session_table"

# ================= CSV VALIDATION =================
csv_files = list(DATA_DIR.glob("*.csv"))

if not csv_files:
    logger.warning("⚠ No CSV files found in data/ directory")
    logger.info("Please add your CSV files to the 'data' folder")
    logger.info("System will still start and wait for data files")
    csv_files = []
else:
    file_metadata = {
        f.name: {
            "size_mb": round(f.stat().st_size / (1024 * 1024), 3),
            "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
        }
        for f in csv_files
    }
    logger.info(f"✓ Found {len(csv_files)} CSV file(s)")
    logger.info(f"File metadata:\n{json.dumps(file_metadata, indent=2)}")

# ================= DATABASE =================
db = SqliteDb(
    db_file=str(BASE_DIR / ModelConfig.DB_FILE),
    session_table=ModelConfig.SESSION_TABLE
)
logger.info("✓ Database initialized")

# ================= MODELS =================
logger.info("Initializing multi-model system...")

discovery_model = Ollama(id=ModelConfig.DISCOVERY_MODEL)
analysis_model = Ollama(id=ModelConfig.ANALYSIS_MODEL)
stats_model = Ollama(id=ModelConfig.STATS_MODEL)
viz_model = Ollama(id=ModelConfig.VIZ_MODEL)
leader_model = Ollama(id=ModelConfig.LEADER_MODEL)

logger.info("✓ All models initialized!")

# ================= AGENTS =================

# 1. DATA DISCOVERY AGENT
data_discovery_agent = Agent(
    id="data-discovery-agent",
    name="Data Discovery Agent",
    model=discovery_model,
    db=db,
    role="Quick file discovery and metadata extraction",
    instructions=[
        "Scan the data/ directory for CSV files.",
        "Return: filename, size, row count, column count, column names.",
        "Use CsvTools to get detailed information.",
        "Keep responses concise and structured.",
    ],
    tools=[
        CsvTools(csvs=csv_files) if csv_files else None,
        FileTools(base_dir=str(BASE_DIR))
    ],
    add_history_to_context=True,
    markdown=False
)

# 2. DATA ANALYSIS AGENT
data_analysis_agent = Agent(
    id="data-analysis-agent",
    name="Data Analysis Agent",
    model=analysis_model,
    db=db,
    role="Complex data analysis and transformation",
    instructions=[
        "🚨 CRITICAL: Use ONLY PandasTools - NO matplotlib, NO plt, NO custom plotting!",
        "",
        "YOUR WORKFLOW:",
        "1. Load CSV: PandasTools.create_dataframe_from_csv(csv_path='data/filename.csv', df_name='df')",
        "2. Inspect: PandasTools.get_dataframe_info(df_name='df')",
        "3. Statistics: PandasTools.get_column_statistics(df_name='df')",
        "4. Return text summary only",
        "",
        "❌ FORBIDDEN: matplotlib, seaborn, plt.show(), any plotting code",
        "✅ ALLOWED: Only PandasTools methods",
    ],
    tools=[
        PandasTools(),
        CsvTools(csvs=csv_files) if csv_files else None,
        FileTools(base_dir=str(BASE_DIR))
    ],
    add_history_to_context=True,
    num_history_runs=10
)

# 3. STATISTICAL AGENT
statistical_agent = Agent(
    id="statistical-agent",
    name="Statistical Analysis Agent",
    model=stats_model,
    db=db,
    role="Statistical analysis using PandasTools only",
    instructions=[
        "🚨 Use ONLY PandasTools - NO matplotlib!",
        "",
        "Available operations:",
        "• PandasTools.get_correlation_matrix(df_name='df')",
        "• PandasTools.get_value_counts(df_name='df', column='col')",
        "• PandasTools.get_column_statistics(df_name='df')",
        "",
        "Return statistical insights as TEXT only.",
        "",
        "❌ FORBIDDEN: Any plotting or visualization code",
        "✅ ALLOWED: Only statistical analysis via PandasTools",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=str(BASE_DIR))
    ],
    add_history_to_context=True,
    num_history_runs=8
)

# 4. VISUALIZATION AGENT
visualization_agent = Agent(
    id="visualization-agent",
    name="Visualization Agent",
    model=viz_model,
    db=db,
    role="Create visualizations using VisualizationTools ONLY",
    instructions=[
        "🚨 CRITICAL: Use ONLY VisualizationTools - NO matplotlib imports!",
        "",
        "=" * 70,
        "MANDATORY TOOL USAGE - USE THESE EXACT METHODS:",
        "=" * 70,
        "",
        "1. HISTOGRAM:",
        "   VisualizationTools.create_histogram(",
        "     data=df['column'],",
        "     output_path='plots/histogram.png',",
        "     title='Title',",
        "     xlabel='X', ylabel='Y'",
        "   )",
        "",
        "2. BAR CHART:",
        "   VisualizationTools.create_bar_chart(",
        "     categories=['A', 'B', 'C'],",
        "     values=[10, 20, 30],",
        "     output_path='plots/bar.png',",
        "     title='Title',",
        "     xlabel='X', ylabel='Y'",
        "   )",
        "",
        "3. SCATTER PLOT:",
        "   VisualizationTools.create_scatter_plot(",
        "     x=df['col1'], y=df['col2'],",
        "     output_path='plots/scatter.png',",
        "     title='Title',",
        "     xlabel='X', ylabel='Y'",
        "   )",
        "",
        "4. BOX PLOT:",
        "   VisualizationTools.create_box_plot(",
        "     data=df['column'],",
        "     output_path='plots/boxplot.png',",
        "     title='Title', ylabel='Y'",
        "   )",
        "",
        "=" * 70,
        "FILE NAMING:",
        "=" * 70,
        "✅ CORRECT: 'plots/price_distribution.png'",
        "❌ WRONG: '/full/path/plots/chart.png', 'chart.png'",
        "",
        "=" * 70,
        "REQUIRED VISUALIZATIONS:",
        "=" * 70,
        "Create exactly 5 charts based on the dataset.",
        "After creating each, list the file path.",
        "",
        "❌ ABSOLUTELY FORBIDDEN:",
        "  - import matplotlib",
        "  - import seaborn",
        "  - plt.figure(), plt.show()",
        "  - Any custom plotting code",
        "",
        "✅ ONLY USE: VisualizationTools methods listed above",
    ],
    tools=[
        PandasTools(),
        VisualizationTools(output_dir=str(PLOTS_DIR)),
        FileTools(base_dir=str(BASE_DIR))
    ],
    add_history_to_context=True,
    markdown=True
)

# 5. REPORT AGENT
report_agent = Agent(
    id="report-agent",
    name="Report Generation Agent",
    model=viz_model,
    db=db,
    role="Generate comprehensive markdown reports",
    instructions=[
        "🚨 YOU MUST USE FileTools.write_file() TO SAVE THE REPORT",
        "",
        "=" * 70,
        "STEP-BY-STEP PROCESS:",
        "=" * 70,
        "",
        "STEP 1: Build report content as string with this structure:",
        "",
        "```",
        "# Data Analysis Report",
        "",
        "*Generated: {timestamp}*",
        "",
        "## Executive Summary",
        "• Key Finding 1",
        "• Key Finding 2",
        "• Key Finding 3",
        "",
        "## Data Overview",
        "- Dataset: filename.csv",
        "- Records: X rows",
        "- Columns: Y columns",
        "",
        "## Statistical Analysis",
        "[Insert statistics]",
        "",
        "## Visualizations Created",
        "1. Chart 1 - `plots/chart1.png`",
        "2. Chart 2 - `plots/chart2.png`",
        "...",
        "",
        "## Recommendations",
        "1. Recommendation 1",
        "2. Recommendation 2",
        "",
        "---",
        "*End of Report*",
        "```",
        "",
        "STEP 2: Save using FileTools:",
        "```python",
        "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')",
        "FileTools.write_file(",
        "  path=f'reports/analysis_report_{timestamp}.md',",
        "  content=report_content",
        ")",
        "```",
        "",
        "STEP 3: In your response, state the file path clearly.",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=str(BASE_DIR))
    ],
    add_history_to_context=True,
    markdown=True
)

# ================= TEAM LEADER =================
data_science_team = Team(
    id="multi-model-data-science-team",
    name="Multi-Model Data Science Team",
    model=leader_model,
    members=[
        data_discovery_agent,
        data_analysis_agent,
        statistical_agent,
        visualization_agent,
        report_agent
    ],
    role="Strategic Data Science Team Leader",
    instructions=[
        "You are the team leader coordinating a complete data science workflow.",
        "",
        "=" * 70,
        "EXECUTION SEQUENCE - FOLLOW EXACTLY:",
        "=" * 70,
        "",
        "PHASE 1: DISCOVERY",
        "→ Delegate to data-discovery-agent",
        "→ Get file metadata and column information",
        "",
        "PHASE 2: DATA LOADING & ANALYSIS",
        "→ Delegate to data-analysis-agent",
        "→ Load CSV and get basic statistics",
        "→ NO VISUALIZATION in this phase",
        "",
        "PHASE 3: STATISTICAL ANALYSIS",
        "→ Delegate to statistical-agent",
        "→ Get correlations, distributions, value counts",
        "→ NO VISUALIZATION in this phase",
        "",
        "PHASE 4: VISUALIZATION (CRITICAL)",
        "→ Delegate to visualization-agent",
        "→ Agent MUST create exactly 5 charts using VisualizationTools",
        "→ Collect all chart file paths from response",
        "",
        "PHASE 5: REPORT GENERATION",
        "→ Delegate to report-agent",
        "→ Agent MUST use FileTools.write_file()",
        "→ Get report file path from response",
        "",
        "PHASE 6: FINAL RESPONSE",
        "",
        "After all phases, provide a summary like this:",
        "",
        "```",
        "📊 Analysis Complete!",
        "",
        "**Key Findings:**",
        "• [Finding 1]",
        "• [Finding 2]",
        "• [Finding 3]",
        "",
        "**Generated Files:**",
        "",
        "Visualizations:",
        "• plots/chart1.png",
        "• plots/chart2.png",
        "• plots/chart3.png",
        "• plots/chart4.png",
        "• plots/chart5.png",
        "",
        "Report:",
        "• reports/analysis_report_TIMESTAMP.md",
        "",
        "All files are saved in your VS Code workspace.",
        "```",
        "",
        "=" * 70,
        "CRITICAL RULES:",
        "=" * 70,
        "1. Each agent uses ONLY their assigned tools",
        "2. NO matplotlib code in data-analysis-agent or statistical-agent",
        "3. ONLY visualization-agent creates charts",
        "4. ONLY report-agent writes markdown files",
        "5. Provide clear file paths in final response",
    ],
    db=db,
    add_history_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True,
    num_history_runs=15,
    markdown=True
)

# ================= AGENT OS =================
agent_os = AgentOS(
    id="multi-model-agent-os",
    name="Multi-Model Data Science Assistant",
    description=(
        "Complete data science system with multi-model orchestration:\n"
        f"• Leader: {ModelConfig.LEADER_MODEL}\n"
        f"• Analysis: {ModelConfig.ANALYSIS_MODEL}\n"
        f"• Statistics: {ModelConfig.STATS_MODEL}\n"
        f"• Visualization: {ModelConfig.VIZ_MODEL}\n"
        f"• Discovery: {ModelConfig.DISCOVERY_MODEL}\n\n"
        "Upload CSV files to the 'data' folder and ask me to analyze them!"
    ),
    teams=[data_science_team]
)

# ================= MAIN =================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Multi-Model Data Science System Starting...")
    logger.info("=" * 70)
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"Plots Directory: {PLOTS_DIR}")
    logger.info(f"Reports Directory: {REPORTS_DIR}")
    logger.info(f"CSV Files Found: {len(csv_files)}")
    if csv_files:
        logger.info(f"Available files: {[f.name for f in csv_files]}")
    logger.info("=" * 70)
    logger.info("Starting web server on http://localhost:7777")
    logger.info("=" * 70)
    
    app = agent_os.get_app()
    agent_os.serve(
        app=app,
        host="localhost",
        port=7777
    )
