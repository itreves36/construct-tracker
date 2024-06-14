import pandas as pd


def create_binary_dataset(df_metadata, dv="suicide", n_per_dv=3000):
    df_metadata_tag_1 = df_metadata[df_metadata[dv] == 1].sample(n=n_per_dv, random_state=123)
    df_metadata_tag_0 = df_metadata[df_metadata[dv] == 0].sample(n=n_per_dv, random_state=123)
    assert df_metadata_tag_1.shape[0] == n_per_dv
    assert df_metadata_tag_0.shape[0] == n_per_dv

    df_metadata_tag = pd.concat([df_metadata_tag_1, df_metadata_tag_0]).reset_index(drop=True)

    return df_metadata_tag


category = "suicidewatch"
not_category = "depression"
suicide_depression = df[df["subreddit"].isin([category, not_category])].reset_index(drop=True)
suicide_depression
