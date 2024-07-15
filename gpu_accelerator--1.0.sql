CREATE FUNCTION gpu_scan(anyelement) 
     RETURNS SETOF record
AS 'MODULE_PATHNAME', 'gpu_scan'
LANGUAGE C STRICT;

CREATE FUNCTION gpu_join(anyelement) 
     RETURNS SETOF record
AS 'MODULE_PATHNAME', 'gpu_join'
LANGUAGE C STRICT;