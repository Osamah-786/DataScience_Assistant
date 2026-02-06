"""
Multi-Model Advanced Data Science System for Agno Platform
Uses different specialized LLMs for different tasks to optimize performance

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

from dotenv import load_dotenv
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
class ModelConfig:
    """Multi-model configuration for specialized tasks"""
    
    # Team Leader - Strong reasoning for orchestration
    LEADER_MODEL = "llama3.1:8b"
    
    # Data Analysis - Largest model for complex computations
    ANALYSIS_MODEL = "qwen2.5:14b-instruct"
    
    # Statistical Analysis - Good at mathematical reasoning
    STATS_MODEL = "mistral:7b-instruct"
    
    # Visualization - Medium model for chart generation
    VIZ_MODEL = "qwen2.5:7b-instruct"
    
    # Data Discovery - Fast lightweight model for simple tasks
    DISCOVERY_MODEL = "llama3.2:3b"
    
    # Database
    DB_FILE = "memory.db"
    SESSION_TABLE = "session_table"
    
    @classmethod
    def get_all_models(cls):
        return [
            cls.LEADER_MODEL,
            cls.ANALYSIS_MODEL,
            cls.STATS_MODEL,
            cls.VIZ_MODEL,
            cls.DISCOVERY_MODEL
        ]

# ================= DATA VALIDATION =================
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    raise FileNotFoundError("No CSV files found in data/")

file_metadata = {
    f.name: {
        "size_mb": round(f.stat().st_size / (1024 * 1024), 3),
        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
    }
    for f in csv_files
}

logger.info(f"Found {len(csv_files)} CSV file(s)")
logger.info(f"File metadata:\n{json.dumps(file_metadata, indent=2)}")

# ================= DB =================
db = SqliteDb(
    db_file=ModelConfig.DB_FILE,
    session_table=ModelConfig.SESSION_TABLE
)
logger.info("Database initialized")

# ================= MODELS =================
# Initialize different models for different tasks
logger.info("Initializing multi-model system for Agno Platform...")

# Model 1: Fast lightweight model for discovery
discovery_model = Ollama(id=ModelConfig.DISCOVERY_MODEL)
logger.info(f"✓ Discovery Model: {ModelConfig.DISCOVERY_MODEL}")

# Model 2: Powerful model for complex data analysis
analysis_model = Ollama(id=ModelConfig.ANALYSIS_MODEL)
logger.info(f"✓ Analysis Model: {ModelConfig.ANALYSIS_MODEL}")

# Model 3: Mathematical reasoning for statistics
stats_model = Ollama(id=ModelConfig.STATS_MODEL)
logger.info(f"✓ Statistics Model: {ModelConfig.STATS_MODEL}")

# Model 4: Medium model for visualizations
viz_model = Ollama(id=ModelConfig.VIZ_MODEL)
logger.info(f"✓ Visualization Model: {ModelConfig.VIZ_MODEL}")

# Model 5: Strong reasoning for team leadership
leader_model = Ollama(id=ModelConfig.LEADER_MODEL)
logger.info(f"✓ Team Leader Model: {ModelConfig.LEADER_MODEL}")

logger.info("All models initialized successfully!")

# ================= AGENTS =================

# 1. DATA DISCOVERY AGENT (Fast & Efficient)
data_discovery_agent = Agent(
    id="data-discovery-agent",
    name="Data Discovery Agent",
    model=discovery_model,
    db=db,
    role="Quick file discovery and metadata extraction",
    instructions=[
        "You provide quick file metadata and basic information.",
        "Be concise and direct - no explanations, just facts.",
        "",
        "When asked about files, return:",
        "- File name(s)",
        "- Size (in MB or KB)",
        "- Row count",
        "- Column count",
        "- Column names",
        "",
        "Format: Clean, structured output without commentary.",
        "Example: 'car_details.csv: 334KB, 8,128 rows, 13 columns (name, year, price, ...)'",
        "",
        f"Available files: {', '.join([f.name for f in csv_files])}",
    ],
    tools=[
        CsvTools(csvs=csv_files),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    markdown=False  # Keep responses plain
)

# 2. DATA ANALYSIS AGENT (Most Powerful)
data_analysis_agent = Agent(
    id="data-analysis-agent",
    name="Data Analysis Agent",
    model=analysis_model,  # Powerful 14B model
    db=db,
    role="Complex data analysis and transformation",
    instructions=[
        "You are the MOST POWERFUL analyst with a 14B parameter model.",
        "Handle complex computations, transformations, and analysis.",
        "",
        "EXECUTION PROTOCOL:",
        "1. Create dataframe: PandasTools.create_dataframe_from_csv",
        "   - Path format: data/filename.csv",
        "   - Dataframe name: car_df (or appropriate name)",
        "",
        "2. Perform complex analysis:",
        "   - Multi-level groupby operations",
        "   - Advanced aggregations",
        "   - Feature engineering (create new columns)",
        "   - Data cleaning (handle nulls, duplicates)",
        "   - Complex filtering with multiple conditions",
        "   - Pivot tables and reshaping",
        "",
        "3. Best practices:",
        "   - ALWAYS use operation_parameters={}",
        "   - Chain operations efficiently",
        "   - Optimize for performance",
        "   - Validate results",
        "",
        "4. Return comprehensive results with insights.",
    ],
    tools=[
        PandasTools(),
        CsvTools(csvs=csv_files),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=10
)

# 3. STATISTICAL ANALYSIS AGENT (Mathematical Expert)
statistical_agent = Agent(
    id="statistical-agent",
    name="Statistical Analysis Agent",
    model=stats_model,  # Mistral 7B for math
    db=db,
    role="Advanced statistical analysis and hypothesis testing",
    instructions=[
        "You are a statistical expert with strong mathematical reasoning.",
        "",
        "STATISTICAL CAPABILITIES:",
        "1. Descriptive Statistics:",
        "   - Mean, median, mode, std, variance",
        "   - Quartiles, percentiles, IQR",
        "   - Skewness and kurtosis",
        "   - Range and coefficient of variation",
        "",
        "2. Correlation Analysis:",
        "   - Pearson correlation",
        "   - Spearman correlation",
        "   - Correlation matrices",
        "   - Statistical significance",
        "",
        "3. Outlier Detection:",
        "   - IQR method",
        "   - Z-score method",
        "   - Modified Z-score",
        "   - Visual identification",
        "",
        "4. Distribution Analysis:",
        "   - Normality tests",
        "   - Distribution fitting",
        "   - Q-Q plots interpretation",
        "",
        "5. Trend Analysis:",
        "   - Linear trends",
        "   - Moving averages",
        "   - Pattern recognition",
        "",
        "Always explain statistical findings in business terms.",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=8
)

# 4. VISUALIZATION AGENT (Chart Specialist)
visualization_agent = Agent(
    id="visualization-agent",
    name="Visualization Agent",
    model=viz_model,  # Qwen 7B for charts
    db=db,
    role="Professional data visualization and chart generation",
    instructions=[
        "You are a data visualization expert creating publication-quality charts.",
        "",
        "CHART TYPES BY USE CASE:",
        "1. Distributions:",
        "   - Histograms (show frequency distribution)",
        "   - Box plots (show quartiles and outliers)",
        "   - Violin plots (density + distribution)",
        "   - KDE plots (smooth distribution)",
        "",
        "2. Comparisons:",
        "   - Bar charts (compare categories)",
        "   - Grouped bars (multi-category comparison)",
        "   - Horizontal bars (long category names)",
        "",
        "3. Relationships:",
        "   - Scatter plots (two variables)",
        "   - Bubble charts (three variables)",
        "   - Correlation heatmaps (many variables)",
        "",
        "4. Time Series:",
        "   - Line charts (trends over time)",
        "   - Area charts (cumulative trends)",
        "",
        "5. Proportions:",
        "   - Pie charts (parts of whole)",
        "   - Donut charts (modern alternative)",
        "   - Stacked bars (composition)",
        "",
        "QUALITY STANDARDS:",
        "- Clear, descriptive titles",
        "- Labeled axes with units",
        "- Appropriate color schemes",
        "- Legends when needed",
        "- Grid lines for readability",
        "- Professional styling",
        "",
        "Save all plots to plots/ directory with descriptive filenames.",
    ],
    tools=[
        PandasTools(),
        VisualizationTools(output_dir=str(PLOTS_DIR)),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True
)

# 5. REPORT GENERATION AGENT (Documentation Expert)
report_agent = Agent(
    id="report-agent",
    name="Report Generation Agent",
    model=viz_model,  # Reuse Qwen 7B for reports
    db=db,
    role="Comprehensive report writing and documentation",
    instructions=[
        "You create professional, comprehensive data analysis reports.",
        "",
        "REPORT STRUCTURE:",
        "# Executive Summary",
        "- Key findings (3-5 bullet points)",
        "- Critical insights",
        "- Recommendations",
        "",
        "# Data Overview",
        "- Dataset description",
        "- Number of records",
        "- Time period covered",
        "- Data quality notes",
        "",
        "# Detailed Analysis",
        "- Statistical summaries",
        "- Trends and patterns",
        "- Correlations",
        "- Outliers and anomalies",
        "",
        "# Visualizations",
        "- Reference to generated charts",
        "- Interpretation of each chart",
        "",
        "# Statistical Insights",
        "- Hypothesis testing results",
        "- Confidence intervals",
        "- Statistical significance",
        "",
        "# Recommendations",
        "- Actionable next steps",
        "- Areas for further investigation",
        "- Business implications",
        "",
        "# Appendix",
        "- Methodology",
        "- Data sources",
        "- Limitations",
        "",
        "Format: Professional Markdown",
        "Save to: reports/ directory with timestamp",
        "Filename format: report_YYYY-MM-DD_HH-MM-SS.md",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True
)

# ================= TEAM =================
data_science_team = Team(
    id="multi-model-data-science-team",
    name="Multi-Model Data Science Team",
    model=leader_model,  # Llama 3.1 8B for leadership
    members=[
        data_discovery_agent,     # Llama 3.2 3B
        data_analysis_agent,      # Qwen 2.5 14B
        statistical_agent,        # Mistral 7B
        visualization_agent,      # Qwen 2.5 7B
        report_agent             # Qwen 2.5 7B
    ],
    role="Strategic Data Science Team Leader",
    instructions=[
        "You are a SILENT TEAM LEADER. You coordinate agents behind the scenes.",
        "",
        "🚨 CRITICAL RULES - READ CAREFULLY:",
        "1. NEVER explain your delegation process to the user",
        "2. NEVER say 'I will delegate' or 'To answer this question'",
        "3. NEVER show internal reasoning or planning",
        "4. Delegate tasks silently using tools",
        "5. Wait for agent responses",
        "6. Present ONLY the final result to the user",
        "7. Speak as if YOU did the work, not the agents",
        "",
        "🎯 TEAM COMPOSITION:",
        "- data-discovery-agent (Llama 3.2 3B) - File metadata, row counts",
        "- data-analysis-agent (Qwen 2.5 14B) - Complex computations",
        "- statistical-agent (Mistral 7B) - Statistical tests",
        "- visualization-agent (Qwen 2.5 7B) - Charts",
        "- report-agent (Qwen 2.5 7B) - Reports",
        "",
        "📋 DELEGATION MAPPING:",
        "Files/metadata/schema → data-discovery-agent",
        "Data analysis/groupby/aggregation → data-analysis-agent",
        "Statistics/correlations → statistical-agent",
        "Charts/plots → visualization-agent",
        "Reports → report-agent",
        "",
        "✅ CORRECT RESPONSE PATTERN:",
        "User: 'What files do I have?'",
        "[Silently delegate to data-discovery-agent]",
        "You: 'I found 1 CSV file: car_details.csv (334KB, 8,128 rows, 13 columns)'",
        "",
        "❌ WRONG RESPONSE PATTERN:",
        "User: 'What files do I have?'",
        "You: 'To answer this question, I will delegate to data-discovery-agent...'",
        "",
        "⚡ WORKFLOW:",
        "1. Receive user query",
        "2. Silently identify which agent(s) to use",
        "3. Delegate WITHOUT telling the user",
        "4. Wait for results",
        "5. Present results as YOUR answer",
        "6. Be concise and direct",
        "",
        "Remember: The user doesn't care about your internal process.",
        "They just want the answer. Act like a unified assistant, not a team manager.",
    ],
    db=db,
    add_history_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True,
    num_history_runs=15,
    markdown=False  # Disable markdown in responses if available
)

# ================= AGENT OS =================
agent_os = AgentOS(
    id="multi-model-agent-os",
    name="Multi-Model Data Science Assistant",
    description=(
        "Advanced multi-model data science system using specialized LLMs:\n"
        f"- Leader: {ModelConfig.LEADER_MODEL}\n"
        f"- Analysis: {ModelConfig.ANALYSIS_MODEL}\n"
        f"- Statistics: {ModelConfig.STATS_MODEL}\n"
        f"- Visualization: {ModelConfig.VIZ_MODEL}\n"
        f"- Discovery: {ModelConfig.DISCOVERY_MODEL}"
    ),
    teams=[data_science_team]
)

# ================= MAIN =================
# ================= MAIN =================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("Multi-Model Advanced Data Science Assistant for Agno")
    logger.info("=" * 70)
    logger.info("Model Distribution:")
    logger.info(f"  Team Leader    : {ModelConfig.LEADER_MODEL}")
    logger.info(f"  Data Discovery : {ModelConfig.DISCOVERY_MODEL}")
    logger.info(f"  Data Analysis  : {ModelConfig.ANALYSIS_MODEL}")
    logger.info(f"  Statistics     : {ModelConfig.STATS_MODEL}")
    logger.info(f"  Visualization  : {ModelConfig.VIZ_MODEL}")
    logger.info("=" * 70)
    logger.info(f"Base Directory: {BASE_DIR}")
    logger.info(f"Data Directory: {DATA_DIR}")
    logger.info(f"CSV Files: {[f.name for f in csv_files]}")
    logger.info("=" * 70)
    
    # Get the app instance and serve it
    app = agent_os.get_app()
    agent_os.serve(
        app=app,
        host="localhost",
        port=7777
    )




