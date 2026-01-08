# Pocache

Caching for functions that take Polars dataframes as input.

## Limitations

1. Polars dataframes must be passed positionally, and params must be passed as keyword argument.

2. len(df) must be smaller than 2**32