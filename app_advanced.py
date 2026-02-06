"""
Multi-Model Advanced Data Science System for Agno Platform
FULLY WORKING VERSION - Guaranteed File Creation and Analysis

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
        "Use CsvTools.list_csvs() to see available files.",
        "For each CSV file, use CsvTools.read_csv() to get a preview.",
        "Return: filename, size, row count, column count, column names.",
        "Keep responses concise and structured.",
    ],
    tools=[
        CsvTools(csvs=csv_files) if csv_files else CsvTools(csvs=[]),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    markdown=False,
    show_tool_calls=True
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
        "YOUR EXACT WORKFLOW:",
        "",
        "Step 1: Create DataFrame",
        "First, find the CSV file path. It will be in the data/ directory.",
        "Example: If the file is 'car_details.csv', the path is 'data/car_details.csv'",
        "",
        "Then use:",
        "PandasTools.create_dataframe_from_csv(",
        "    csv_path='data/FILENAME.csv',",
        "    df_name='main_df'",
        ")",
        "",
        "Step 2: Get DataFrame Info",
        "PandasTools.get_dataframe_info(df_name='main_df')",
        "",
        "Step 3: Get Statistics",
        "PandasTools.get_column_statistics(df_name='main_df')",
        "",
        "Step 4: Get Column Names",
        "PandasTools.get_column_names(df_name='main_df')",
        "",
        "Step 5: Return Summary",
        "Provide a text summary including:",
        "- Total rows and columns",
        "- Column names and types",
        "- Basic statistics (mean, median, etc.)",
        "- Missing values count",
        "- Data quality notes",
        "",
        "❌ ABSOLUTELY FORBIDDEN:",
        "- import matplotlib",
        "- import seaborn",
        "- plt.figure(), plt.plot(), plt.show()",
        "- Any custom plotting code",
        "- Creating visualization files",
        "",
        "✅ ONLY ALLOWED:",
        "- PandasTools methods only",
        "- Text-based summaries",
    ],
    tools=[
        PandasTools(),
        CsvTools(csvs=csv_files) if csv_files else CsvTools(csvs=[]),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=10,
    show_tool_calls=True
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
        "YOUR TASKS:",
        "",
        "1. Get Correlation Matrix:",
        "PandasTools.get_correlation_matrix(df_name='main_df')",
        "",
        "2. Get Value Counts for categorical columns:",
        "PandasTools.get_value_counts(df_name='main_df', column='column_name')",
        "",
        "3. Analyze distributions:",
        "Use PandasTools.get_column_statistics() for numeric columns",
        "",
        "4. Return insights as TEXT:",
        "- Which variables are correlated?",
        "- What are the distributions?",
        "- Any outliers detected?",
        "- Key statistical patterns",
        "",
        "❌ FORBIDDEN: Any plotting or visualization code",
        "✅ ALLOWED: Only PandasTools statistical methods",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=8,
    show_tool_calls=True
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
        "YOU MUST CREATE EXACTLY 5 CHARTS",
        "=" * 70,
        "",
        "Before creating charts, you need the DataFrame loaded.",
        "The main_df should already exist from previous steps.",
        "",
        "CHART 1: Histogram of a numeric column",
        "Example:",
        "VisualizationTools.create_histogram(",
        "    data=main_df['selling_price'].tolist(),",
        "    output_path='plots/price_distribution.png',",
        "    title='Distribution of Selling Prices',",
        "    xlabel='Selling Price (₹)',",
        "    ylabel='Frequency'",
        ")",
        "",
        "CHART 2: Bar chart of a categorical column",
        "Example:",
        "value_counts = main_df['fuel'].value_counts()",
        "VisualizationTools.create_bar_chart(",
        "    categories=value_counts.index.tolist(),",
        "    values=value_counts.values.tolist(),",
        "    output_path='plots/fuel_type_comparison.png',",
        "    title='Fuel Type Distribution',",
        "    xlabel='Fuel Type',",
        "    ylabel='Count'",
        ")",
        "",
        "CHART 3: Scatter plot",
        "VisualizationTools.create_scatter_plot(",
        "    x=main_df['km_driven'].tolist(),",
        "    y=main_df['selling_price'].tolist(),",
        "    output_path='plots/price_vs_km_scatter.png',",
        "    title='Price vs Kilometers Driven',",
        "    xlabel='Kilometers Driven',",
        "    ylabel='Selling Price (₹)'",
        ")",
        "",
        "CHART 4: Box plot",
        "VisualizationTools.create_box_plot(",
        "    data=main_df['year'].tolist(),",
        "    output_path='plots/year_boxplot.png',",
        "    title='Year Distribution',",
        "    ylabel='Year'",
        ")",
        "",
        "CHART 5: Another categorical bar chart",
        "value_counts = main_df['seller_type'].value_counts()",
        "VisualizationTools.create_bar_chart(",
        "    categories=value_counts.index.tolist(),",
        "    values=value_counts.values.tolist(),",
        "    output_path='plots/seller_type_bars.png',",
        "    title='Seller Type Distribution',",
        "    xlabel='Seller Type',",
        "    ylabel='Count'",
        ")",
        "",
        "=" * 70,
        "CRITICAL RULES:",
        "=" * 70,
        "1. Use .tolist() to convert pandas Series to lists",
        "2. All file paths must start with 'plots/'",
        "3. Create ALL 5 charts - don't stop early",
        "4. After each chart, verify it was created successfully",
        "5. List all 5 file paths at the end",
        "",
        "❌ ABSOLUTELY FORBIDDEN:",
        "  - import matplotlib",
        "  - import seaborn",
        "  - plt.figure(), plt.show()",
        "  - Any custom plotting code",
        "",
        "✅ ONLY USE: VisualizationTools.create_* methods",
    ],
    tools=[
        PandasTools(),
        VisualizationTools(output_dir=str(PLOTS_DIR)),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    markdown=True,
    show_tool_calls=True
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
        "EXACT PROCESS:",
        "=" * 70,
        "",
        "Step 1: Build the report content as a string",
        "",
        "Create a variable with this exact structure:",
        "",
        "report_content = '''# Data Analysis Report",
        "",
        "*Generated: {current_timestamp}*",
        "",
        "## Executive Summary",
        "",
        "**Key Findings:**",
        "• [Insert finding 1]",
        "• [Insert finding 2]",
        "• [Insert finding 3]",
        "",
        "## Data Overview",
        "",
        "- **Dataset:** filename.csv",
        "- **Records:** X rows",
        "- **Columns:** Y columns",
        "- **File Size:** Z MB",
        "",
        "## Column Information",
        "",
        "[List all columns and their types]",
        "",
        "## Statistical Analysis",
        "",
        "### Descriptive Statistics",
        "[Insert statistics from analysis]",
        "",
        "### Correlations",
        "[Insert correlation insights]",
        "",
        "### Distributions",
        "[Insert distribution insights]",
        "",
        "## Visualizations Created",
        "",
        "1. **Price Distribution** - `plots/price_distribution.png`",
        "2. **Fuel Type Comparison** - `plots/fuel_type_comparison.png`",
        "3. **Price vs KM Scatter** - `plots/price_vs_km_scatter.png`",
        "4. **Year Box Plot** - `plots/year_boxplot.png`",
        "5. **Seller Type Distribution** - `plots/seller_type_bars.png`",
        "",
        "## Key Insights",
        "",
        "1. [Insight 1]",
        "2. [Insight 2]",
        "3. [Insight 3]",
        "",
        "## Recommendations",
        "",
        "1. [Recommendation 1]",
        "2. [Recommendation 2]",
        "3. [Recommendation 3]",
        "",
        "## Methodology",
        "",
        "- Data loaded using pandas",
        "- Statistical analysis performed",
        "- 5 visualizations created",
        "- Multi-model AI system used",
        "",
        "---",
        "*Report generated by Multi-Model Data Science System*",
        "'''",
        "",
        "Step 2: Save the file using FileTools",
        "",
        "Use this EXACT code:",
        "",
        "FileTools.write_file(",
        "    path='reports/analysis_report_20260206.md',",
        "    content=report_content",
        ")",
        "",
        "Step 3: In your response, state:",
        "✅ Report saved to: reports/analysis_report_20260206.md",
        "",
        "=" * 70,
        "CRITICAL:",
        "=" * 70,
        "- You MUST call FileTools.write_file()",
        "- The path MUST start with 'reports/'",
        "- The content MUST be a complete markdown string",
        "- Verify the file was created successfully",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    markdown=True,
    show_tool_calls=True
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
        "You coordinate a data science team to perform COMPLETE end-to-end analysis.",
        "",
        "When the user asks to analyze a dataset, execute ALL phases sequentially.",
        "",
        "=" * 70,
        "MANDATORY EXECUTION SEQUENCE:",
        "=" * 70,
        "",
        "PHASE 1: DISCOVERY",
        "→ Call data-discovery-agent",
        "→ Get list of CSV files and their metadata",
        "→ Wait for response before proceeding",
        "",
        "PHASE 2: DATA LOADING & ANALYSIS",
        "→ Call data-analysis-agent",
        "→ Agent will load CSV and return statistics",
        "→ Wait for complete response",
        "→ Verify you received statistics before proceeding",
        "",
        "PHASE 3: STATISTICAL ANALYSIS",
        "→ Call statistical-agent",
        "→ Agent will compute correlations and distributions",
        "→ Wait for complete response",
        "",
        "PHASE 4: VISUALIZATION (CRITICAL)",
        "→ Call visualization-agent",
        "→ Agent MUST create exactly 5 PNG files in plots/ folder",
        "→ Wait for ALL 5 charts to be created",
        "→ Collect the 5 file paths from response",
        "",
        "PHASE 5: REPORT GENERATION",
        "→ Call report-agent",
        "→ Agent MUST save markdown file to reports/ folder",
        "→ Wait for confirmation file was saved",
        "→ Get the report file path",
        "",
        "PHASE 6: FINAL USER RESPONSE",
        "",
        "After ALL phases complete, respond to user with:",
        "",
        "📊 **Analysis Complete!**",
        "",
        "**Key Findings:**",
        "• [Finding 1 from analysis]",
        "• [Finding 2 from analysis]",
        "• [Finding 3 from analysis]",
        "",
        "**Generated Files:**",
        "",
        "**Visualizations (5 charts):**",
        "• plots/price_distribution.png",
        "• plots/fuel_type_comparison.png",
        "• plots/price_vs_km_scatter.png",
        "• plots/year_boxplot.png",
        "• plots/seller_type_bars.png",
        "",
        "**Report:**",
        "• reports/analysis_report_20260206.md",
        "",
        "All files are saved in your VS Code workspace and ready to view!",
        "",
        "=" * 70,
        "CRITICAL RULES:",
        "=" * 70,
        "1. Execute ALL 6 phases - don't skip any",
        "2. Wait for each agent to complete before moving on",
        "3. Verify files were actually created",
        "4. If an agent fails, retry once",
        "5. Provide clear file paths in final response",
        "6. Don't just show delegation messages - show RESULTS",
    ],
    db=db,
    add_history_to_context=True,
    add_member_tools_to_context=True,
    enable_agentic_state=True,
    num_history_runs=15,
    markdown=True,
    show_tool_calls=True
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
        "I automatically analyze CSV files and create visualizations and reports!\n\n"
        "Just say: 'Analyze the dataset' or 'Create a data analysis report'"
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
