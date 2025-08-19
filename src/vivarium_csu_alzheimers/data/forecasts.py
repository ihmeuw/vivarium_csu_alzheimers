import vivarium
import xarray as xr, numpy as np, pandas as pd


def table_from_nc(fname_dict, param, loc_id, loc_name, age_mapping):
    ds = xr.open_dataset(
        fname_dict[param],
        engine="netcdf4",  # let xarray auto-detect; list here if you know it
        decode_cf=True,  # handle CF-conventions (time units, etc.)
    )

    # Select relevant part of FHS dataset
    var_name = param
    if param == "births":
        var_name = "population"
    elif param in ["migration", "mortality"]:
        var_name = "value"

    pop_ts = ds[var_name].sel(
        location_id=loc_id,
    )

    if param != "migration":
        pop_ts = pop_ts.isel(scenario=0)

    pop_ts = pop_ts.squeeze(drop=True)  # remove now-singleton dims

    df = pop_ts.to_dataframe(name="value").reset_index()

    # Transform to vivarium format
    # 1. Convert location_id to location name
    df["location"] = loc_name

    # 2. Convert sex_id to sex names
    sex_mapping = {
        1: "Male",
        2: "Female",
    }
    df["sex"] = df["sex_id"].map(sex_mapping)

    # 3. Convert age_group_id to age intervals
    if param != "births":
        age_bins = age_mapping.set_index("age_group_id")
        df["age_start"] = np.round(df["age_group_id"].map(age_bins["age_start"]), 3)
        df["age_end"] = np.round(df["age_group_id"].map(age_bins["age_end"]), 3)
        age_cols = ["age_start", "age_end"]
    else:
        age_cols = []

    # 4. Convert year_id to year intervals
    df["year_start"] = df["year_id"].map(int)
    df["year_end"] = df["year_id"].map(int) + 1

    # 5. Set index and unstack to get draw columns
    index_cols = (
        [
            "location",
            "sex",
        ]
        + age_cols
        + ["year_start", "year_end", "draw"]
    )
    df_indexed = df.dropna(subset=index_cols).set_index(index_cols)

    df_wide = df_indexed["value"].unstack(level="draw")

    # 6. Rename columns to draw_x format
    df_wide.columns = [f"draw_{col}" for col in df_wide.columns]

    return df_wide
