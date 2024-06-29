"""
Data Visualization & Preprocessing
"""
from typing import Union, Dict
import pandas as pd
import sweetviz

def auto_eda(data: Union[pd.DataFrame, Dict]) -> None:
    """
    Using sweetviz for automation exploratory data analysis (EDA)
    sweetviz: https://pypi.org/project/sweetviz/
    
    data: (pd.DataFrame / Dict) data to be analyzed
    """
    report = sweetviz.analyze(data)
    report.show_html(
        filepath="eda_report.html",
        open_browser=False,
        layout="widescreen",
        scale=0.8
    )
