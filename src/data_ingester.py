"""Data ingestion module for CKD survival model development
This module handles the ingestion of data from various sources of CSV files
"""

import os
import glob
import pandas as pd
import numpy as np
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple

class DataIngester(ABC):
    """Base class for data ingesters"""
    
    @abstractmethod
    def ingest_data(self):
        """Method to ingest data"""
        pass

class CSVDataIngester(DataIngester):
    """Ingests data from CSV files, output a raw dataframe"""
    
    def __init__(self, directory: str, file_pattern: str = "*.csv", header_row: int = 0):
        """
        Initialize the CSV data ingester.
        
        Args:
            directory: Directory containing CSV files
            file_pattern: Glob pattern to match CSV files (default: "*.csv")
            header_row: Row to use as the header (default: 0)
                        For ICD10 files, use header_row=1 to skip the first row with numeric indices
        """
        self.directory = directory
        self.file_pattern = file_pattern
        self.header_row = header_row

    def ingest_data(self) -> List[pd.DataFrame]:
        """
        Ingest data from all CSV files in the specified directory and return a list of DataFrames.
        
        Returns:
            List of DataFrames, one for each CSV file
        """
        csv_files = glob.glob(os.path.join(self.directory, self.file_pattern))
        data_frames = []
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.directory} matching pattern {self.file_pattern}")
        
        for file in csv_files:
            try:
                df = pd.read_csv(file, header=self.header_row, low_memory=False)
                
                # Add metadata
                filename = os.path.basename(file)
                df['source_file'] = filename
                
                # Extract year and quarter information from filename
                year_match = re.search(r'(\d{4})', filename)
                quarter_match = re.search(r'q([1-4])', filename.lower())
                
                if year_match:
                    df['year'] = year_match.group(1)
                
                if quarter_match:
                    df['quarter'] = quarter_match.group(1)
                
                data_frames.append(df)
                print(f"Loaded {file} with {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not data_frames:
            raise ValueError(f"Failed to load any CSV files from {self.directory}")
        
        return data_frames
    
    def ingest(self) -> pd.DataFrame:
        """
        Ingest data from all CSV files in the specified directory and return a combined DataFrame.
        
        Returns:
            Combined DataFrame with data from all CSV files
        """
        data_frames = self.ingest_data()
        
        if not data_frames:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(data_frames, ignore_index=True)
        print(f"Combined {len(data_frames)} files with a total of {len(combined_df)} rows")
        
        return combined_df