import sys
import io
import json
import os
import glob

from ScienceParse import ScienceParse


def read_science_parse(paperid, title, abstract, scienceparse_dir):
    pdf_path = os.path.join(scienceparse_dir, "{}.pdf.json".format(paperid))
    scienceparse_file = io.open(pdf_path, "r", encoding="utf8")
    scienceparse_str = scienceparse_file.read()
    scienceparse_data = json.loads(scienceparse_str)

    # read scienceparse
    sections = {}
    reference_years = {}
    reference_titles = {}
    reference_venues = {}
    reference_mention_contexts = {}
    reference_num_mentions = {}

    name = scienceparse_data["name"]
    metadata = scienceparse_data["metadata"]

    if metadata["sections"] is not None:
        for sectid in range(len(metadata["sections"])):
            heading = metadata["sections"][sectid]["heading"]
            text = metadata["sections"][sectid]["text"]
            sections[str(heading)] = text

    for refid in range(len(metadata["references"])):
        reference_titles[refid] = metadata["references"][refid]["title"]
        reference_years[refid] = metadata["references"][refid]["year"]
        reference_venues[refid] = metadata["references"][refid]["venue"]

    for menid in range(len(metadata["referenceMentions"])):
        refid = metadata["referenceMentions"][menid]["referenceID"]
        context = metadata["referenceMentions"][menid]["context"]
        oldContext = reference_mention_contexts.get(refid, "")
        reference_mention_contexts[refid] = oldContext + "\t" + context
        count = reference_num_mentions.get(refid, 0)
        reference_num_mentions[refid] = count + 1

    authors = metadata["authors"]
    emails = metadata["emails"]

    science_parse = ScienceParse(title, abstract, sections, reference_titles, reference_venues, reference_years,
                                 reference_mention_contexts, reference_num_mentions, authors, emails)
    return science_parse
