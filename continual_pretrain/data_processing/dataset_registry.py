
from typing import Dict, Callable, Optional, Union, List, Tuple


class DatasetFormatterRegistry:
    """Enhanced registry supporting single or multiple column processing."""
    
    def __init__(self):
        # Store (formatter_function, columns_to_process) tuples
        self._formatters: Dict[str, Tuple[Callable, Union[str, List[str]]]] = {}
    
    def register(self, dataset_name: str, formatter: Callable, 
                 columns: Union[str, List[str]] = "messages"):
        """
        Register a formatter for a specific dataset.
        
        Args:
            dataset_name: Identifier for the dataset
            formatter: Function that formats the data
            columns: Column name(s) to process (str or list of str)
        """
        self._formatters[dataset_name] = (formatter, columns)
        col_display = columns if isinstance(columns, str) else ", ".join(columns)
        print(f"✅ Registered formatter for '{dataset_name}' (columns: {col_display})")
    
    def get_formatter(self, dataset_name: str) -> Optional[Tuple[Callable, Union[str, List[str]]]]:
        """Get the formatter and column name(s) for a dataset."""
        return self._formatters.get(dataset_name)
    
    def list_datasets(self):
        """List all registered datasets with their columns."""
        result = []
        for name, (_, cols) in self._formatters.items():
            col_display = cols if isinstance(cols, str) else ", ".join(cols)
            result.append((name, col_display))
        return result
    
    def format_dataset(self, dataset, dataset_name: str, 
                      tokenizer=None, user_token="<utilizator>", 
                      assistant_token="<asistent>", system_token="<sistem>", 
                      num_proc=1):
        """Format a dataset using the registered formatter."""
        formatter_info = self.get_formatter(dataset_name)
        
        if formatter_info is None:
            available = self.list_datasets()
            raise ValueError(
                f"No formatter found for '{dataset_name}'. "
                f"Available: {[name for name, _ in available]}"
            )
        
        formatter_func, columns = formatter_info
        
        # Handle single column or multiple columns
        if isinstance(columns, str):
            columns = [columns]
        
        # Verify all columns exist
        for col in columns:
            if col not in dataset.column_names:
                raise ValueError(
                    f"Column '{col}' not found in dataset. "
                    f"Available columns: {dataset.column_names}"
                )
        
        def format_example(example):
            try:
                # Extract data from specified column(s)
                if len(columns) == 1:
                    data_to_format = example[columns[0]]
                else:
                    # For multiple columns, pass a dict
                    data_to_format = {col: example[col] for col in columns}
                
                # Format using the registered function
                example['formatted_text'] = formatter_func(
                    data_to_format,
                    tokenizer=tokenizer,
                    user_token=user_token,
                    assistant_token=assistant_token,
                    system_token=system_token
                )
            except Exception as e:
                print(f"⚠️  Error formatting example: {e}")
                example['formatted_text'] = ""
            
            return example
        
        return dataset.map(
            format_example,
            num_proc=num_proc,
            desc=f"Formatting {dataset_name}"
        )
