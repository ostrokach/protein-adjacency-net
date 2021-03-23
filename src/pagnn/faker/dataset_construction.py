import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_mismatch_mutations(df_ref: pd.DataFrame) -> pd.DataFrame:
    """

    This function is used to generate the humsavar validation dataset.
    """
    assert df_ref["domain_start"].min() >= 1
    assert df_ref["q_start"].min() >= 1

    df = df_ref.copy()

    df["domain_mutation"] = df.apply(_get_domain_mutation, axis=1)
    df["qseq_mutation"] = df.apply(_get_qseq_mutation, axis=1)

    df["mutation_in_domain"] = df.apply(_check_mutation_in_domain, axis=1)
    logger.info(
        f"Removing {(~df['mutation_in_domain']).sum()} rows "
        f"containing mutations that are outside domains."
    )
    df = df[df["mutation_in_domain"]]

    df["mutation_matches_sequence"] = df.apply(_check_mutation_matches_sequence, axis=1)
    logger.info(
        f"Removing {(~df['mutation_matches_sequence']).sum()} rows "
        f"containing mutations do not match sequence."
    )
    df = df[df["mutation_matches_sequence"]]

    df["mutation_in_alignment"] = df.apply(_check_mutation_in_alignment, axis=1)
    logger.info(
        f"Removing {(~df['mutation_in_alignment']).sum()} rows "
        f"containing mutations that are outside alignment with structure."
    )
    df = df[df["mutation_in_alignment"]]

    df["mutation_matches_qseq"] = df.apply(_check_mutation_matches_qseq, axis=1)
    assert df["mutation_matches_qseq"].all()

    columns_to_drop = [
        "mutation_in_domain",
        "mutation_matches_sequence",
        "mutation_in_alignment",
        "mutation_matches_qseq",
    ]
    df = df.drop(pd.Index(columns_to_drop), axis=1)

    return df


def _get_domain_mutation(s):
    return (
        f"{s.uniprot_mutation[0]}"
        f"{int(s.uniprot_mutation[1:-1]) - int(s.domain_start - 1)}"
        f"{s.uniprot_mutation[-1]}"
    )


def _get_qseq_mutation(s):
    return (
        f"{s.domain_mutation[0]}"
        f"{int(s.domain_mutation[1:-1]) - int(s.q_start - 1)}"
        f"{s.domain_mutation[-1]}"
    )


def _check_mutation_in_domain(s):
    return s.domain_start <= int(s.uniprot_mutation[1:-1]) <= s.domain_end


def _check_mutation_matches_sequence(s):
    mutation_pos = int(s.domain_mutation[1:-1])
    assert mutation_pos > 0
    if len(s.sequence) < mutation_pos:
        return False
    return s.sequence[mutation_pos - 1] == s.domain_mutation[0]


def _check_mutation_in_alignment(s):
    return s.q_start <= int(s.domain_mutation[1:-1]) <= s.q_end


def _check_mutation_matches_qseq(s):
    mutation_pos = int(s.qseq_mutation[1:-1])
    assert mutation_pos > 0
    qseq = s.qseq.replace("-", "")
    if len(qseq) < mutation_pos:
        return False
    return qseq[mutation_pos - 1] == s.qseq_mutation[0]
