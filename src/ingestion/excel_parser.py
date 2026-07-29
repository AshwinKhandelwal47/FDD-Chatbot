import logging
import pandas as pd
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

def extract_tables_from_excel(file_path: Path) -> list[dict[str, Any]]:
    """Extract tables from an Excel or CSV file.
    
    Each returned dict contains the shared metadata schema:
        - data: list[dict] (records)
        - headers: list[str] (column names)
        - page_number: str (sheet name, or 'CSV' for csv files)
        - table_index: int
        - source_file: str
        - chunk_type: 'table'
        - section: str | None
        - fiscal_year: int | None
        - extraction_method: str ('pandas_excel' or 'pandas_csv')
    """
    tables: list[dict[str, Any]] = []
    
    try:
        if file_path.suffix.lower() == ".csv":
            df = pd.read_csv(file_path)
            # Drop completely empty rows
            df = df.dropna(how='all')
            # Fill NA values to avoid non-serializable float NaNs in JSON
            df = df.fillna("")
            
            headers = [str(c) for c in df.columns]
            records = df.to_dict(orient="records")
            
            tables.append({
                "page_number": "CSV",
                "table_index": 1,
                "data": records,
                "headers": headers,
                "source_file": str(file_path.name),
                "chunk_type": "table",
                "section": None,
                "fiscal_year": None,
                "extraction_method": "pandas_csv",
            })
        else:
            # Excel file
            sheet_dict = pd.read_excel(file_path, sheet_name=None)
            for sheet_index, (sheet_name, df) in enumerate(sheet_dict.items(), start=1):
                df = df.dropna(how='all')
                df = df.fillna("")
                
                if df.empty and len(df.columns) == 0:
                    continue
                    
                headers = [str(c) for c in df.columns]
                records = df.to_dict(orient="records")
                
                tables.append({
                    "page_number": sheet_name,
                    "table_index": 1,
                    "data": records,
                    "headers": headers,
                    "source_file": str(file_path.name),
                    "chunk_type": "table",
                    "section": None,
                    "fiscal_year": None,
                    "extraction_method": "pandas_excel",
                })
    except Exception as exc:
        logger.error(
            "Failed to extract tables from '%s': %s", file_path, exc
        )
        raise

    return tables
