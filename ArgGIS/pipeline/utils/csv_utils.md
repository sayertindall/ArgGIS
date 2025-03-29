# CSV Utilities

This module provides a set of utility functions for handling CSV files in Python. These functions are designed to streamline common tasks associated with CSV data manipulation and analysis.
Functions

1. `load_csv()`: Loads a CSV file into a pandas DataFrame with proper error handling and logging.

2. `preview()`: Shows the first few rows of a DataFrame for quick inspection.

3. `validate_columns()`: Checks if a DataFrame contains all required columns.

4. `clean_column_names()`: Standardizes column names by removing spaces, converting to lowercase, and replacing special characters.

5. `drop_empty_columns()`: Removes columns that have more than a specified percentage of missing values.

6. `find_duplicate_rows()`: Identifies and returns duplicate rows in a DataFrame.

7. `export_to_csv()`: Saves a DataFrame to a CSV file with error handling.

8. `summarize_columns()`: Generates a summary of columns including data types, unique values, and missing value percentages.

9. `merge_csv_files()`: Combines multiple CSV files into a single DataFrame using a specified join method.

10. `filter_by_value()`: Filters a DataFrame based on a condition applied to a specific column.

11. `pivot_table()`: Creates a pivot table from a DataFrame with specified dimensions and aggregation function.

12. `detect_outliers()`: Identifies rows containing outlier values in a numeric column using various statistical methods.

13. `find_missing_values()`: Analyzes and summarizes missing values across all columns in a DataFrame.

14. `analyze_excel_file()`: Provides a detailed analysis of Excel file structure and content.

15. `smart_type_converter()`: Automatically detects and converts column data types based on content analysis.

16. `standardize_excel()`: Reformats Excel files with consistent formatting, column names, and structure.

17. `deduplicate_with_fuzzy()`: Removes duplicate records using fuzzy string matching for text similarity.

18. `auto_map_columns()`: Automatically maps columns between dataframes based on name similarity.

19. `hierarchical_normalizer()`: Normalizes hierarchical data with parent-child relationships across multiple levels.

20. `distribution_matcher()`: Transforms values to match a target statistical distribution while preserving rank order.

21. `auto_categorizer()`: Creates optimal categories from continuous data using statistical methods.

22. `seasonal_normalizer()`: Removes seasonal patterns from time series data for proper trend analysis.

23. `smart_outlier_normalizer()`: Detects and handles outliers using context-aware methods based on data distribution.

24. `cross_column_normalizer()`: Normalizes values across related columns to ensure consistency.

25. `context_aware_text_normalizer()`: Performs text normalization with locale-specific rules for better standardization.

26. `peek_all_sheets()`: Previews all sheets in an Excel file without assuming header structure.

27. `analyze_sheet_differences()`: Compares sheet names and structures across multiple Excel files.

28. `analyze_header_patterns()`: Evaluates header row patterns across Excel files.

29. `generate_updated_manifest()`: Creates a manifest file with correct sheet mappings based on file analysis.

30. `create_sheet_mapping_report()`: Produces a detailed report on sheet name variations across Excel files.

31. `preview_file_structure()`: Examines Excel file structure including sheet layout and formatting.

32. `interactive_excel_explorer()`: Provides comprehensive information about Excel file structure and content.

33. `explore_excel_files()`: Lists all sheets in Excel files within a directory.

34. `print_sheet_names_by_year()`: Displays sheet names grouped by year for pattern recognition.

35. `build_sheet_mapping()`: Creates a mapping between sheet names and their likely content type.

36. `create_fixed_manifest()`: Generates a manifest file based on actual Excel sheet data.

37. `preview_excel_content()`: Shows sample content from all sheets in an Excel file.
