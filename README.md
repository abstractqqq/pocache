# Pocache

Caching for functions that take Polars dataframes as input. I do not intend to publish this as a package
because you can just copy it.

## Limitations

1. Polars dataframes must be passed positionally, and params must be passed as keyword argument.

2. len(df) must be smaller than 2**32