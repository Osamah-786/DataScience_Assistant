# """
# Advanced Multi-Agent Data Science System
# Optimized for Agno Platform Deployment

# Features:
# - Multi-agent orchestration
# - Advanced analytics
# - Data visualization
# - Statistical insights
# - Report generation
# """

# from agno.agent import Agent
# from agno.models.ollama import Ollama
# from agno.db.sqlite import SqliteDb
# from agno.tools.csv_toolkit import CsvTools
# from agno.tools.file import FileTools
# from agno.tools.pandas import PandasTools
# from agno.tools.visualization import VisualizationTools
# from agno.team import Team
# from agno.os import AgentOS

# from dotenv import load_dotenv
# from pathlib import Path
# from typing import List, Dict, Any
# from datetime import datetime
# import logging
# import json

# # ================= LOGGING CONFIGURATION =================
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('agent_system.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # ================= ENVIRONMENT SETUP =================
# load_dotenv()

# BASE_DIR = Path(__file__).parent
# DATA_DIR = BASE_DIR / "data"
# PLOTS_DIR = BASE_DIR / "plots"
# REPORTS_DIR = BASE_DIR / "reports"
# CACHE_DIR = BASE_DIR / "cache"

# # Create directories if they don't exist
# for directory in [DATA_DIR, PLOTS_DIR, REPORTS_DIR, CACHE_DIR]:
#     directory.mkdir(exist_ok=True)

# # ================= CONFIGURATION =================
# class Config:
#     """Centralized configuration management"""
    
#     # Model settings
#     MODEL_NAME = "qwen2.5:3b-instruct"
    
#     # Database settings
#     DB_FILE = "memory.db"
#     SESSION_TABLE = "session_table"
    
#     # Agent settings
#     CONTEXT_HISTORY_RUNS = 10
#     MAX_TOOL_CALLS = 50
    
#     @classmethod
#     def to_dict(cls) -> Dict[str, Any]:
#         return {k: v for k, v in cls.__dict__.items() if not k.startswith('_')}

# # ================= UTILITIES =================
# class DataValidator:
#     """Validate CSV files and data integrity"""
    
#     @staticmethod
#     def validate_csv_files(data_dir: Path) -> List[Path]:
#         """Validate and return CSV files"""
#         csv_files = list(data_dir.glob("*.csv"))
        
#         if not csv_files:
#             logger.error(f"No CSV files found in {data_dir}")
#             raise FileNotFoundError(f"No CSV files found in {data_dir}")
        
#         logger.info(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")
#         return csv_files
    
#     @staticmethod
#     def get_file_metadata(csv_files: List[Path]) -> Dict[str, Any]:
#         """Get metadata about CSV files"""
#         metadata = {}
#         for csv_file in csv_files:
#             metadata[csv_file.name] = {
#                 'size_mb': csv_file.stat().st_size / (1024 * 1024),
#                 'modified': datetime.fromtimestamp(csv_file.stat().st_mtime).isoformat()
#             }
#         return metadata

# # ================= INITIALIZE SYSTEM =================
# logger.info("Initializing Advanced Data Science Agent System for Agno Platform")

# # Validate data files
# csv_files = DataValidator.validate_csv_files(DATA_DIR)
# file_metadata = DataValidator.get_file_metadata(csv_files)
# logger.info(f"File metadata: {json.dumps(file_metadata, indent=2)}")

# # ================= DATABASE =================
# db = SqliteDb(
#     db_file=Config.DB_FILE,
#     session_table=Config.SESSION_TABLE
# )
# logger.info("Database initialized")

# # ================= MODEL =================
# model = Ollama(id=Config.MODEL_NAME)
# logger.info(f"Model initialized: {Config.MODEL_NAME}")

# # ================= AGENT DEFINITIONS =================

# # 1. DATA DISCOVERY AGENT
# data_discovery_agent = Agent(
#     id="data-discovery-agent",
#     name="Data Discovery Agent",
#     model=model,
#     db=db,
#     role="Data source discovery and validation",
#     instructions=[
#         "You are a data discovery specialist.",
#         "Your responsibilities:",
#         "1. Identify and validate CSV files in the data/ directory",
#         "2. Provide file metadata (size, columns, row counts)",
#         "3. Suggest data quality checks",
#         "4. DO NOT perform pandas operations - only discovery",
#         "",
#         "Available files:",
#         *[f"- {f.name} ({f.stat().st_size / 1024:.1f} KB)" for f in csv_files],
#     ],
#     tools=[
#         CsvTools(csvs=csv_files),
#         FileTools(base_dir=BASE_DIR)
#     ],
#     add_history_to_context=True
# )

# # 2. DATA ANALYSIS AGENT
# data_analysis_agent = Agent(
#     id="data-analysis-agent",
#     name="Data Analysis Agent",
#     model=model,
#     db=db,
#     role="Advanced data analysis and transformation",
#     instructions=[
#         "You are an expert data analyst and executor.",
#         "",
#         "EXECUTION PROTOCOL:",
#         "1. Create dataframe: PandasTools.create_dataframe_from_csv",
#         "   - Use exact path from data/ folder",
#         "   - Standard naming: {filename}_df (e.g., car_details.csv → car_df)",
#         "",
#         "2. Execute requested analysis immediately:",
#         "   - Statistical summaries (describe, groupby, aggregations)",
#         "   - Data cleaning (handle nulls, duplicates, outliers)",
#         "   - Feature engineering (new columns, transformations)",
#         "   - Filtering and sorting",
#         "   - Correlation analysis",
#         "",
#         "3. Validation:",
#         "   - Always verify dataframe exists before operations",
#         "   - Check for data quality issues",
#         "   - Report any anomalies",
#         "",
#         "4. Return structured results:",
#         "   - Clear tables or summaries",
#         "   - Key insights",
#         "   - Recommendations if applicable",
#         "",
#         "BEST PRACTICES:",
#         "- Use operation_parameters={} for all pandas operations",
#         "- Chain operations when logical",
#         "- Provide context with numerical results",
#         "- Flag missing or invalid data",
#     ],
#     tools=[
#         PandasTools(),
#         CsvTools(csvs=csv_files),
#         FileTools(base_dir=BASE_DIR)
#     ],
#     add_history_to_context=True,
#     num_history_runs=Config.CONTEXT_HISTORY_RUNS
# )

# # 3. VISUALIZATION AGENT
# visualization_agent = Agent(
#     id="visualization-agent",
#     name="Visualization Agent",
#     model=model,
#     db=db,
#     role="Advanced data visualization and plotting",
#     instructions=[
#         "You are a data visualization expert.",
#         "",
#         "VISUALIZATION GUIDELINES:",
#         "1. Chart selection:",
#         "   - Distributions → histograms, box plots, violin plots",
#         "   - Comparisons → bar charts, grouped bars",
#         "   - Relationships → scatter plots, correlation heatmaps",
#         "   - Time series → line charts, area charts",
#         "   - Categorical → pie charts, stacked bars",
#         "",
#         "2. Quality standards:",
#         "   - Always add descriptive titles",
#         "   - Label axes clearly with units",
#         "   - Use color palettes thoughtfully",
#         "   - Add legends when needed",
#         "   - Include grid for readability",
#         "",
#         "3. Output:",
#         "   - Save to plots/ directory",
#         "   - Use descriptive filenames",
#         "   - Support multiple formats (PNG, SVG)",
#         "",
#         "4. Prerequisites:",
#         "   - Verify dataframe exists before plotting",
#         "   - Check column names and data types",
#         "   - Handle missing data appropriately",
#     ],
#     tools=[
#         PandasTools(),
#         VisualizationTools(output_dir=str(PLOTS_DIR)),
#         FileTools(base_dir=BASE_DIR)
#     ],
#     add_history_to_context=True
# )

# # 4. STATISTICAL INSIGHTS AGENT
# statistical_agent = Agent(
#     id="statistical-insights-agent",
#     name="Statistical Insights Agent",
#     model=model,
#     db=db,
#     role="Statistical analysis and hypothesis testing",
#     instructions=[
#         "You are a statistical analysis expert.",
#         "",
#         "CAPABILITIES:",
#         "1. Descriptive statistics:",
#         "   - Central tendency (mean, median, mode)",
#         "   - Dispersion (std, variance, range, IQR)",
#         "   - Distribution shape (skewness, kurtosis)",
#         "",
#         "2. Inferential statistics:",
#         "   - Correlation analysis",
#         "   - Trend detection",
#         "   - Outlier identification",
#         "",
#         "3. Reporting:",
#         "   - Statistical summaries with interpretation",
#         "   - Identify significant patterns",
#         "   - Suggest further analysis",
#         "",
#         "Always provide context and business insights with statistical results.",
#     ],
#     tools=[
#         PandasTools(),
#         FileTools(base_dir=BASE_DIR)
#     ],
#     add_history_to_context=True
# )

# # 5. REPORT GENERATION AGENT
# report_agent = Agent(
#     id="report-generation-agent",
#     name="Report Generation Agent",
#     model=model,
#     db=db,
#     role="Comprehensive report creation",
#     instructions=[
#         "You generate professional data analysis reports.",
#         "",
#         "REPORT STRUCTURE:",
#         "1. Executive Summary",
#         "2. Data Overview",
#         "3. Key Findings",
#         "4. Visualizations",
#         "5. Statistical Insights",
#         "6. Recommendations",
#         "7. Appendix",
#         "",
#         "Format: Markdown with embedded references to plots and tables.",
#         "Save reports to reports/ directory with timestamp.",
#     ],
#     tools=[
#         FileTools(base_dir=BASE_DIR),
#         PandasTools()
#     ],
#     add_history_to_context=True
# )

# # ================= TEAM CONFIGURATION =================
# data_science_team = Team(
#     id="advanced-data-science-team",
#     name="Advanced Data Science Team",
#     model=model,
#     members=[
#         data_discovery_agent,
#         data_analysis_agent,
#         visualization_agent,
#         statistical_agent,
#         report_agent
#     ],
#     role="Senior Data Science Orchestrator",
#     instructions=[
#         "You are the lead data scientist coordinating a specialized team.",
#         "",
#         "🎯 ORCHESTRATION PRINCIPLES:",
#         "",
#         "1. TASK DECOMPOSITION:",
#         "   - Break complex requests into logical subtasks",
#         "   - Identify which specialist handles each subtask",
#         "   - Plan execution order (dependencies matter)",
#         "",
#         "2. AGENT ROUTING:",
#         "   - File discovery/validation → data-discovery-agent",
#         "   - Data loading, cleaning, analysis → data-analysis-agent",
#         "   - Charts, plots, graphs → visualization-agent",
#         "   - Statistical tests, correlations → statistical-insights-agent",
#         "   - Final reports, summaries → report-generation-agent",
#         "",
#         "3. EXECUTION MANAGEMENT:",
#         "   - Delegate tasks with clear, specific instructions",
#         "   - Wait for agent completion before proceeding",
#         "   - Collect and verify results from each agent",
#         "   - Handle errors gracefully (retry or escalate)",
#         "",
#         "4. QUALITY ASSURANCE:",
#         "   - Ensure all requested analyses are completed",
#         "   - Verify outputs are coherent and complete",
#         "   - Synthesize multi-agent results",
#         "   - Provide comprehensive final response to user",
#         "",
#         "5. USER COMMUNICATION:",
#         "   - Explain what you're doing at each step",
#         "   - Present results clearly and professionally",
#         "   - Offer follow-up suggestions",
#         "",
#         "⚠️ CRITICAL RULES:",
#         "- Never stop after delegation - always collect results",
#         "- Never invent data - only report actual outputs",
#         "- Always maintain context across delegations",
#         "- Prioritize accuracy over speed",
#     ],
#     db=db,
#     add_history_to_context=True,
#     add_member_tools_to_context=True,
#     enable_agentic_state=True,
#     num_history_runs=Config.CONTEXT_HISTORY_RUNS
# )

# # ================= AGENT OS =================
# agent_os = AgentOS(
#     id="advanced-agent-os",
#     name="Advanced Data Science Assistant",
#     description=(
#         "Enterprise-grade multi-agent data science system with advanced analytics, "
#         "visualization, statistical insights, and automated reporting capabilities"
#     ),
#     teams=[data_science_team]
# )

# # ================= MAIN =================
# if __name__ == "__main__":
#     logger.info("=" * 60)
#     logger.info("Advanced Data Science Agent System")
#     logger.info("Ready for Agno Platform Deployment")
#     logger.info("=" * 60)
#     logger.info(f"Base Directory: {BASE_DIR}")
#     logger.info(f"Data Directory: {DATA_DIR}")
#     logger.info(f"CSV Files: {[f.name for f in csv_files]}")
#     logger.info("=" * 60)
    
#     # For Agno platform, just serve the AgentOS
#     # The platform will handle the web interface
#     agent_os.serve(app="app:agent_os")