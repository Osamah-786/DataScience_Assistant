"""
Multi-Model Advanced Data Science System for Agno Platform
Uses different specialized LLMs for different tasks to optimize performance

Models:
- llama3.1:8b → Team Leader (orchestration)
- qwen2.5:14b-instruct → Data Analysis (complex computations)
- mistral:7b-instruct → Statistical Analysis (mathematical reasoning)
- qwen2.5:7b-instruct → Data Discovery & Reports
- Gemini Pro (API) → Visualization (chart generation with Google AI)

NEW FEATURES:
- Chat History Management
- Conversation Export
- Session Tracking
- Analysis History Search
"""

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.pandas import PandasTools
from agno.team import Team
from agno.os import AgentOS
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import logging
import json
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()

# ================= LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ================= ENV =================
load_dotenv()

# Validate Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables!\n"
        "Please add GEMINI_API_KEY=your_key_here to your .env file\n"
        "Get your free API key from: https://makersuite.google.com/app/apikey"
    )

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PLOTS_DIR = BASE_DIR / "plots"
REPORTS_DIR = BASE_DIR / "reports"
CACHE_DIR = BASE_DIR / "cache"
HISTORY_DIR = BASE_DIR / "chat_history"  # NEW: Chat history storage

for d in [DATA_DIR, PLOTS_DIR, REPORTS_DIR, CACHE_DIR, HISTORY_DIR]:
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
    
    # Data Discovery & Reports - Medium model
    DISCOVERY_MODEL = "qwen2.5:7b-instruct"
    
    # Visualization - Gemini Pro (FREE tier) for advanced chart generation
    # FREE tier limits: 15 RPM, 1M tokens/day, 1500 requests/day
    VIZ_MODEL = "models/gemini-1.0-pro"
  # This is the FREE model
    
    # Database
    DB_FILE = "memory.db"
    SESSION_TABLE = "session_table"
    CHAT_HISTORY_TABLE = "chat_history"  # NEW: Chat history table
    
    @classmethod
    def get_ollama_models(cls):
        return [
            cls.LEADER_MODEL,
            cls.ANALYSIS_MODEL,
            cls.STATS_MODEL,
            cls.DISCOVERY_MODEL
        ]
    
    @classmethod
    def get_all_models(cls):
        return cls.get_ollama_models() + [cls.VIZ_MODEL]

# ================= CHAT HISTORY MANAGER =================
class ChatHistoryManager:
    """Manages chat history storage, retrieval, and export"""
    
    def __init__(self, db_file, history_dir):
        self.db_file = db_file
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(exist_ok=True)
        self._init_history_table()
    
    def _init_history_table(self):
        """Initialize chat history table if it doesn't exist"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_message TEXT,
                agent_response TEXT,
                agent_used TEXT,
                dataset_analyzed TEXT,
                files_generated TEXT,
                metadata TEXT
            )
        """)
        
        # Create index for faster searches
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_timestamp 
            ON chat_history(session_id, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_dataset 
            ON chat_history(dataset_analyzed)
        """)
        
        conn.commit()
        conn.close()
        logger.info("Chat history table initialized")
    
    def save_interaction(self, session_id, user_message, agent_response, 
                        agent_used=None, dataset_analyzed=None, 
                        files_generated=None, metadata=None):
        """Save a chat interaction to history"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chat_history 
            (session_id, user_message, agent_response, agent_used, 
             dataset_analyzed, files_generated, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id,
            user_message,
            agent_response,
            agent_used,
            dataset_analyzed,
            json.dumps(files_generated) if files_generated else None,
            json.dumps(metadata) if metadata else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_session_history(self, session_id, limit=50):
        """Retrieve chat history for a specific session"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, timestamp, user_message, agent_response, 
                   agent_used, dataset_analyzed, files_generated, metadata
            FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        history = []
        for row in results:
            history.append({
                'id': row[0],
                'timestamp': row[1],
                'user_message': row[2],
                'agent_response': row[3],
                'agent_used': row[4],
                'dataset_analyzed': row[5],
                'files_generated': json.loads(row[6]) if row[6] else None,
                'metadata': json.loads(row[7]) if row[7] else None
            })
        
        return history
    
    def search_history(self, query, session_id=None, dataset=None, limit=20):
        """Search chat history by query, session, or dataset"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        sql = """
            SELECT id, session_id, timestamp, user_message, agent_response,
                   agent_used, dataset_analyzed
            FROM chat_history
            WHERE (user_message LIKE ? OR agent_response LIKE ?)
        """
        params = [f'%{query}%', f'%{query}%']
        
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        
        if dataset:
            sql += " AND dataset_analyzed = ?"
            params.append(dataset)
        
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'session_id': row[1],
                'timestamp': row[2],
                'user_message': row[3],
                'agent_response': row[4],
                'agent_used': row[5],
                'dataset_analyzed': row[6]
            }
            for row in results
        ]
    
    def export_session(self, session_id, format='json'):
        """Export session history to JSON or Markdown"""
        history = self.get_session_history(session_id, limit=1000)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            filename = self.history_dir / f"session_{session_id}_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
        
        elif format == 'markdown':
            filename = self.history_dir / f"session_{session_id}_{timestamp}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"# Chat History - Session {session_id}\n\n")
                f.write(f"*Exported: {datetime.now().isoformat()}*\n\n")
                f.write("---\n\n")
                
                for i, item in enumerate(reversed(history), 1):
                    f.write(f"## Interaction {i}\n\n")
                    f.write(f"**Time**: {item['timestamp']}\n\n")
                    if item['dataset_analyzed']:
                        f.write(f"**Dataset**: {item['dataset_analyzed']}\n\n")
                    if item['agent_used']:
                        f.write(f"**Agent**: {item['agent_used']}\n\n")
                    
                    f.write(f"### User:\n{item['user_message']}\n\n")
                    f.write(f"### Assistant:\n{item['agent_response']}\n\n")
                    
                    if item.get('files_generated'):
                        f.write(f"**Files Generated**: {', '.join(item['files_generated'])}\n\n")
                    
                    f.write("---\n\n")
        
        logger.info(f"Session exported to {filename}")
        return str(filename)
    
    def get_all_sessions(self):
        """Get list of all unique sessions"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT session_id, 
                   MIN(timestamp) as first_interaction,
                   MAX(timestamp) as last_interaction,
                   COUNT(*) as interaction_count
            FROM chat_history
            GROUP BY session_id
            ORDER BY last_interaction DESC
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'session_id': row[0],
                'first_interaction': row[1],
                'last_interaction': row[2],
                'interaction_count': row[3]
            }
            for row in results
        ]
    
    def get_statistics(self):
        """Get overall chat history statistics"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total interactions
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        stats['total_interactions'] = cursor.fetchone()[0]
        
        # Total sessions
        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM chat_history")
        stats['total_sessions'] = cursor.fetchone()[0]
        
        # Most analyzed datasets
        cursor.execute("""
            SELECT dataset_analyzed, COUNT(*) as count
            FROM chat_history
            WHERE dataset_analyzed IS NOT NULL
            GROUP BY dataset_analyzed
            ORDER BY count DESC
            LIMIT 5
        """)
        stats['top_datasets'] = cursor.fetchall()
        
        # Most used agents
        cursor.execute("""
            SELECT agent_used, COUNT(*) as count
            FROM chat_history
            WHERE agent_used IS NOT NULL
            GROUP BY agent_used
            ORDER BY count DESC
        """)
        stats['agent_usage'] = cursor.fetchall()
        
        conn.close()
        return stats

# Initialize Chat History Manager
chat_history = ChatHistoryManager(
    db_file=ModelConfig.DB_FILE,
    history_dir=HISTORY_DIR
)

# ================= DATA VALIDATION =================
csv_files = list(DATA_DIR.glob("*.csv"))
if not csv_files:
    logger.warning("No CSV files found in data/ directory. Please add CSV files for analysis.")
    sample_data_path = DATA_DIR / "README.txt"
    sample_data_path.write_text(
        "Add your CSV files to this directory for analysis.\n"
        "Supported formats: .csv\n"
        "Example: sales_data.csv, customer_data.csv, etc."
    )
    csv_files = []

file_metadata = {
    f.name: {
        "size_mb": round(f.stat().st_size / (1024 * 1024), 3),
        "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
    }
    for f in csv_files
}

if csv_files:
    logger.info(f"Found {len(csv_files)} CSV file(s)")
    logger.info(f"File metadata:\n{json.dumps(file_metadata, indent=2)}")

# ================= DB =================
db = SqliteDb(
    db_file=ModelConfig.DB_FILE,
    session_table=ModelConfig.SESSION_TABLE
)
logger.info("Database initialized")

# ================= MODELS =================
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

# Model 4: Gemini Pro for advanced visualizations
viz_model = Gemini(id=ModelConfig.VIZ_MODEL, api_key=GEMINI_API_KEY)
logger.info(f"✓ Visualization Model: {ModelConfig.VIZ_MODEL} (Google Gemini - FREE)")

# Model 5: Strong reasoning for team leadership
leader_model = Ollama(id=ModelConfig.LEADER_MODEL)
logger.info(f"✓ Team Leader Model: {ModelConfig.LEADER_MODEL}")

logger.info("All models initialized successfully!")
logger.info("=" * 70)
logger.info("MODEL ARCHITECTURE:")
logger.info(f"  4 Ollama Models (Local): {', '.join(ModelConfig.get_ollama_models())}")
logger.info(f"  1 Gemini Model (Cloud/FREE): {ModelConfig.VIZ_MODEL}")
logger.info("=" * 70)

# ================= CHAT HISTORY AGENT =================
# NEW: Special agent for managing chat history
chat_history_agent = Agent(
    id="chat-history-agent",
    name="Chat History Manager",
    model=discovery_model,
    db=db,
    role="Manage and retrieve chat conversation history",
    instructions=[
        "You manage chat history and help users find past conversations.",
        "",
        "CAPABILITIES:",
        "1. Show recent conversations",
        "2. Search past analyses",
        "3. Export session history",
        "4. Display statistics",
        "",
        "COMMANDS YOU UNDERSTAND:",
        "- 'show history' / 'show chat history' → Display recent interactions",
        "- 'search for [query]' → Search past conversations",
        "- 'export history' → Export current session to file",
        "- 'show statistics' / 'show stats' → Display usage statistics",
        "- 'list sessions' → Show all past sessions",
        "",
        "When showing history, present it in a clean, readable format:",
        "- Group by date",
        "- Show user query → agent response",
        "- Include relevant metadata (dataset, files generated)",
        "",
        "Be helpful and concise in your responses.",
    ],
    tools=[
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    markdown=True
)

# ================= AGENTS ================= 
# (Using the same agent definitions from your original code)

# 1. DATA DISCOVERY AGENT
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
        f"Available files: {', '.join([f.name for f in csv_files]) if csv_files else 'None - waiting for upload'}",
    ],
    tools=[
        CsvTools(csvs=csv_files) if csv_files else None,
        FileTools(base_dir=BASE_DIR)
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
        "You are the MOST POWERFUL analyst with a 14B parameter model.",
        "Handle complex computations, transformations, and analysis.",
        "",
        "EXECUTION PROTOCOL:",
        "1. Create dataframe: PandasTools.create_dataframe_from_csv",
        "   - Path format: data/filename.csv",
        "   - Dataframe name: df (or appropriate name)",
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
        "   - ALWAYS validate data first",
        "   - Check for missing values",
        "   - Identify data types",
        "   - Chain operations efficiently",
        "   - Optimize for performance",
        "",
        "4. Return comprehensive results with insights.",
        "",
        "ANALYSIS TYPES:",
        "- Exploratory Data Analysis (EDA)",
        "- Data profiling and quality assessment",
        "- Feature engineering and transformation",
        "- Aggregation and summarization",
        "- Time series analysis",
        "- Categorical analysis",
    ],
    tools=[
        PandasTools(),
        CsvTools(csvs=csv_files) if csv_files else None,
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=10
)

# 3. STATISTICAL ANALYSIS AGENT
statistical_agent = Agent(
    id="statistical-agent",
    name="Statistical Analysis Agent",
    model=stats_model,
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
        "   - Visual identification recommendations",
        "",
        "4. Distribution Analysis:",
        "   - Normality assessment",
        "   - Distribution characteristics",
        "   - Skewness and kurtosis interpretation",
        "",
        "5. Trend Analysis:",
        "   - Linear trends",
        "   - Moving averages",
        "   - Pattern recognition",
        "",
        "6. Hypothesis Testing:",
        "   - T-tests",
        "   - Chi-square tests",
        "   - ANOVA",
        "",
        "Always explain statistical findings in business terms.",
        "Provide actionable insights from statistical results.",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=8
)

# 4. VISUALIZATION AGENT (Gemini-Powered)
visualization_agent = Agent(
    id="visualization-agent",
    name="Gemini Visualization Agent",
    model=viz_model,
    db=db,
    role="Professional data visualization using Google Gemini AI (FREE tier)",
    instructions=[
        "You create professional data visualizations using Google Gemini AI FREE tier.",
        "",
        "🎨 GEMINI VISUALIZATION (100% FREE):",
        "- Advanced visual understanding",
        "- Smart color palette selection",
        "- Automatic layout optimization",
        "- Context-aware chart recommendations",
        "- FREE tier: 1,500 requests/day (more than enough!)",
        "",
        "📊 VISUALIZATION WORKFLOW:",
        "",
        "STEP 1: Analyze the data context",
        "- Understand the data type and distribution",
        "- Identify the best chart type for the insight",
        "- Consider the audience and purpose",
        "",
        "STEP 2: Generate visualization code",
        "Create Python code using matplotlib/seaborn/plotly:",
        "",
        "```python",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from pathlib import Path",
        "",
        "# Set style",
        "sns.set_style('whitegrid')",
        "plt.rcParams['figure.figsize'] = (10, 6)",
        "plt.rcParams['font.size'] = 10",
        "",
        "# Create the plot",
        "fig, ax = plt.subplots()",
        "# ... plotting code ...",
        "",
        "# Save with high quality",
        f"output_path = Path('{PLOTS_DIR}') / 'chart_name.png'",
        "plt.savefig(output_path, dpi=300, bbox_inches='tight')",
        "plt.close()",
        "",
        "print(f'✓ Saved: {output_path}')",
        "```",
        "",
        "STEP 3: Execute using PandasTools",
        "Use PandasTools.run_python_code() to execute the visualization code",
        "",
        "STEP 4: Verify file creation",
        "Check that the file exists and return the full path",
        "",
        "=" * 60,
        "",
        "CHART TYPES BY USE CASE:",
        "",
        "1. Distributions:",
        "   - Histogram: frequency distribution",
        "   - KDE Plot: smooth distribution",
        "   - Box plot: quartiles and outliers",
        "   - Violin plot: distribution + statistics",
        "",
        "2. Comparisons:",
        "   - Bar chart: category comparison",
        "   - Grouped bars: multi-category",
        "   - Horizontal bars: long labels",
        "   - Heatmap: matrix comparison",
        "",
        "3. Relationships:",
        "   - Scatter plot: correlation",
        "   - Bubble chart: 3 variables",
        "   - Pair plot: multiple relationships",
        "   - Correlation heatmap: all relationships",
        "",
        "4. Time Series:",
        "   - Line chart: trends",
        "   - Area chart: cumulative",
        "   - Seasonal decomposition",
        "",
        "5. Proportions:",
        "   - Pie chart: parts of whole",
        "   - Donut chart: modern alternative",
        "   - Stacked bars: composition",
        "",
        "QUALITY STANDARDS:",
        "✓ Clear, descriptive titles",
        "✓ Labeled axes with units",
        "✓ Professional color schemes",
        "✓ Legends when needed",
        "✓ Grid lines for readability",
        "✓ High DPI (300) for quality",
        "",
        f"All charts MUST be saved to: {PLOTS_DIR}",
        "Use descriptive filenames: price_distribution.png, sales_trend.png, etc.",
    ],
    tools=[
        PandasTools(),
        FileTools(base_dir=BASE_DIR)
    ],
    add_history_to_context=True,
    num_history_runs=5
)

# 5. REPORT GENERATION AGENT
report_agent = Agent(
    id="report-agent",
    name="Report Generation Agent",
    model=discovery_model,
    db=db,
    role="Comprehensive report writing and documentation",
    instructions=[
        "You create comprehensive data analysis reports in Markdown format.",
        "",
        "📝 REPORT GENERATION PROTOCOL:",
        "",
        "STEP 1: Gather all analysis results",
        "STEP 2: Structure the report using the template",
        "STEP 3: Write the report file",
        "STEP 4: Verify and return path",
        "",
        "Include sections for:",
        "- Executive Summary with key findings",
        "- Dataset Overview",
        "- Statistical Analysis",
        "- Visualizations (with interpretations)",
        "- Recommendations",
        "- Methodology",
        "",
        f"Save all reports to: {REPORTS_DIR}",
        "Use format: analysis_report_YYYYMMDD_HHMMSS.md",
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
    name="Multi-Model Data Science Team with Chat History",
    model=leader_model,
    members=[
        chat_history_agent,  # NEW: Add history agent first
        data_discovery_agent,
        data_analysis_agent,
        statistical_agent,
        visualization_agent,
        report_agent
    ],
    role="Strategic Data Science Team Leader with Chat History Management",
    instructions=[
        "You coordinate a specialized team of data science agents WITH CHAT HISTORY.",
        "",
        "=" * 70,
        "NEW FEATURE: CHAT HISTORY",
        "=" * 70,
        "",
        "You can now track and retrieve past conversations!",
        "",
        "History Commands (delegate to chat-history-agent):",
        "- 'show history' → Recent conversations",
        "- 'search [query]' → Find past analyses",
        "- 'export history' → Save session to file",
        "- 'show stats' → Usage statistics",
        "",
        "=" * 70,
        "RESPONSE STYLE",
        "=" * 70,
        "",
        "Present results directly without explaining delegation.",
        "",
        "✓ GOOD: 'I analyzed the dataset and found...'",
        "✗ BAD: 'I will delegate this to data-analysis-agent...'",
        "",
        "=" * 70,
        "TEAM COMPOSITION",
        "=" * 70,
        "",
        "0. chat-history-agent (Qwen 2.5 7B - Ollama)",
        "   → Chat history management, search, export",
        "",
        "1. data-discovery-agent (Qwen 2.5 7B - Ollama)",
        "   → Fast file scanning, metadata, structure",
        "",
        "2. data-analysis-agent (Qwen 2.5 14B - Ollama)",
        "   → Complex computations, transformations",
        "",
        "3. statistical-agent (Mistral 7B - Ollama)",
        "   → Statistical tests, correlations, distributions",
        "",
        "4. visualization-agent (Gemini Pro - FREE)",
        "   → Professional charts with Google AI",
        "",
        "5. report-agent (Qwen 2.5 7B - Ollama)",
        "   → Comprehensive markdown reports",
        "",
        "=" * 70,
        "DELEGATION MAPPING",
        "=" * 70,
        "",
        "User asks about history → chat-history-agent",
        "User asks about files → data-discovery-agent",
        "User wants analysis → data-analysis-agent",
        "User wants statistics → statistical-agent",
        "User wants charts → visualization-agent (GEMINI FREE)",
        "User wants report → report-agent",
        "User wants COMPLETE analysis → ALL agents in sequence",
        "",
        "=" * 70,
        "FINAL RESPONSE FORMAT",
        "=" * 70,
        "",
        "Always include in your response:",
        "",
        "I've completed [task description].",
        "",
        "🔍 Key Findings:",
        "• [Finding 1]",
        "• [Finding 2]",
        "",
        "📊 Generated Outputs:",
        "• [List of files created]",
        "",
        "💡 Tip: Use 'show history' to see past analyses!",
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
    name="Multi-Model Data Science Assistant with Chat History",
    description=(
        "🚀 Advanced Multi-Model Data Science System\n\n"
        "Ollama Models (Local/FREE):\n"
        f"  • Leader: {ModelConfig.LEADER_MODEL}\n"
        f"  • Analysis: {ModelConfig.ANALYSIS_MODEL}\n"
        f"  • Statistics: {ModelConfig.STATS_MODEL}\n"
        f"  • Discovery: {ModelConfig.DISCOVERY_MODEL}\n\n"
        "Cloud Models:\n"
        f"  • Visualization: {ModelConfig.VIZ_MODEL} (Google Gemini - FREE)\n\n"
        "Capabilities:\n"
        "  ✓ Automated data analysis\n"
        "  ✓ Statistical computing\n"
        "  ✓ AI-powered visualizations (FREE Gemini)\n"
        "  ✓ Report generation\n"
        "  ✓ Chat history tracking\n"
        "  ✓ Conversation search & export\n"
        "  ✓ End-to-end workflows\n\n"
        "💰 Cost: $0/month (FREE tier: 1,500 Gemini requests/day)"
    ),
    teams=[data_science_team]
)

# ================= MAIN =================
if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🚀 Multi-Model Advanced Data Science Assistant")
    logger.info("   WITH CHAT HISTORY TRACKING")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Model Distribution:")
    logger.info(f"  🤖 Team Leader    : {ModelConfig.LEADER_MODEL}")
    logger.info(f"  🔍 Data Discovery : {ModelConfig.DISCOVERY_MODEL}")
    logger.info(f"  📊 Data Analysis  : {ModelConfig.ANALYSIS_MODEL}")
    logger.info(f"  📈 Statistics     : {ModelConfig.STATS_MODEL}")
    logger.info(f"  🎨 Visualization  : {ModelConfig.VIZ_MODEL} (Gemini FREE)")
    logger.info(f"  💾 Chat History   : {ModelConfig.DISCOVERY_MODEL}")
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"📁 Base Directory     : {BASE_DIR}")
    logger.info(f"📂 Data Directory     : {DATA_DIR}")
    logger.info(f"📊 Plots Directory    : {PLOTS_DIR}")
    logger.info(f"📄 Reports Directory  : {REPORTS_DIR}")
    logger.info(f"💬 Chat History Dir   : {HISTORY_DIR}")
    
    if csv_files:
        logger.info(f"📋 CSV Files Found    : {[f.name for f in csv_files]}")
    else:
        logger.info("⚠️  No CSV files found - add files to data/ directory")
    
    # Show chat history statistics
    stats = chat_history.get_statistics()
    logger.info("")
    logger.info("💬 Chat History Statistics:")
    logger.info(f"   Total Interactions : {stats['total_interactions']}")
    logger.info(f"   Total Sessions     : {stats['total_sessions']}")
    
    logger.info("=" * 70)
    logger.info("")
    logger.info("Starting AgentOS on http://localhost:7777")
    logger.info("Try commands like:")
    logger.info("  • 'Analyze my data'")
    logger.info("  • 'Show chat history'")
    logger.info("  • 'Export this session'")
    logger.info("  • 'Show statistics'")
    logger.info("")
    logger.info("Press Ctrl+C to stop")
    logger.info("")
    
    # Get the app instance and serve it
    app = agent_os.get_app()
    agent_os.serve(
        app=app,
        host="localhost",
        port=7777
    )
