# Petroleum Citation and Abstention Pilot

The retrieval context assembly now includes:

- source URL
- stable source ID
- PDF page number
- chunk ID in the result object
- explicit abstention when no positive lexical evidence exists

A live query verified page-aware citation output for the Bombay Shelf question. The retrieved context
identified the report, source ID `usgs-of-1986-0080-south-asia`, and page 47.

The fixed query labels were also corrected to include page 47 as a relevant passage; the original page
22 remains relevant but was not the highest-scoring answer passage.

Updated retrieval metrics:

```text
Hit rate@5: 0.800
Recall@5:   0.818
MRR@5:      0.683
```

This remains a retrieval/context test, not a generated-answer faithfulness test. The answer layer
must quote or calculate only from the cited context and must preserve the source units.
