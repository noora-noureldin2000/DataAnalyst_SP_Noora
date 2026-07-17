import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Union
import pandas as pd
import numpy as np
import re

logger = logging.getLogger(__name__)

class StudyContextParser:
    """Parses a structured study brief into a machine-readable context object."""
    def __init__(self, brief: Union[Dict, str, Path]):
        self.brief_source = brief
        self.title = ""
        self.background = ""
        self.methods_summary = ""
        self.aims = []
        self.primary_outcome = ""
        self.secondary_outcomes = []
        self.study_design = "auto"
        self.keywords = []
        
    def parse(self):
        if isinstance(self.brief_source, (str, Path)):
            with open(self.brief_source, 'r', encoding='utf-8') as f:
                data = json.load(f)
        elif isinstance(self.brief_source, dict):
            data = self.brief_source
        else:
            raise ValueError("Brief must be a dictionary, or path to a JSON file.")
            
        self.title = data.get("title", "")
        self.background = data.get("background", "")
        self.methods_summary = data.get("methods_summary", "")
        self.aims = data.get("aims", [])
        self.primary_outcome = data.get("primary_outcome", "")
        self.secondary_outcomes = data.get("secondary_outcomes", [])
        self.study_design = data.get("design", "auto")
        
        text_corpus = f"{self.title} {self.background} {self.methods_summary} {' '.join(self.aims)}".lower()
        self.keywords = re.findall(r'\b\w+\b', text_corpus)
        
        logger.info(f"Parsed Study Context: '{self.title}' with primary outcome '{self.primary_outcome}'")
        return self


class VariableRoleClassifier:
    """Maps every column in the dataset to a formal statistical role."""
    
    def __init__(self, df: pd.DataFrame, context: StudyContextParser):
        self.df = df
        self.context = context
        self.role_map = {}
        
    def classify(self) -> Dict[str, Dict[str, Any]]:
        self.role_map = {}
        
        primary_out = self.context.primary_outcome
        secondary_outs = self.context.secondary_outcomes
        
        for col in self.df.columns:
            dtype = "continuous"
            if pd.api.types.is_numeric_dtype(self.df[col]):
                n_unique = self.df[col].nunique()
                if n_unique <= 2:
                    dtype = "binary"
                elif n_unique <= 10:
                    dtype = "ordinal"
            else:
                dtype = "nominal"
                if self.df[col].nunique() == 2:
                    dtype = "binary"
                    
            n_unique = self.df[col].nunique()
            missing_pct = self.df[col].isna().mean() * 100
            
            col_lower = str(col).lower()
            role = "IV_continuous" if dtype == "continuous" else "IV_categorical"
            
            if col == primary_out:
                role = "DV_primary"
            elif col in secondary_outs:
                role = "DV_secondary"
            elif any(x in col_lower for x in ["time", "surv", "days", "months", "years"]):
                role = "time"
            elif any(x in col_lower for x in ["status", "event", "censor", "dead", "death"]):
                role = "status"
            elif any(x in col_lower for x in ["id", "subject", "patient"]) and n_unique > self.df.shape[0] * 0.8:
                role = "id"
            
            self.role_map[col] = {
                "role": role,
                "dtype": dtype,
                "n_unique": n_unique,
                "missing_pct": missing_pct,
                "auto_detected": True,
                "user_confirmed": False
            }
            
        return self.role_map
        
    def print_classification_table(self):
        df_roles = self.to_dataframe()
        print("\n--- Variable Classification ---")
        print(df_roles.to_string())
        print("-------------------------------\n")
        
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.role_map, orient='index')


class PsychometricScorer:
    """Auto-scores common standardized psychometric tools."""
    
    @staticmethod
    def score(df: pd.DataFrame, instrument: str, subscale_map: Dict, reverse_items: List[str] = None):
        if reverse_items is None:
            reverse_items = []
            
        df_scored = df.copy()
        for subscale, items in subscale_map.items():
            valid_items = [i for i in items if i in df.columns]
            if not valid_items:
                continue
            df_scored[f"{instrument}_{subscale}"] = df_scored[valid_items].sum(axis=1)
            
        return df_scored
