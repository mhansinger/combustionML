#%%
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%
df_all = pd.read_hdf("./data/tables_of_fgm.h5")

in_labels = ["f", "zeta", "pv"]
#out_labels = df_all.columns.drop(in_labels)

# read in the species order
with open("GRI_species_order_lu13", "r") as f:
    # print(species)
    out_labels = f.read().splitlines()

# discretization of new dimension
points_new_dim = 10

# append other fields: heatrelease,  T, PVs
# labels.append('heatRelease')
out_labels.append("T")
out_labels.append("PVs")

print("The labels are:")
print(out_labels)

out_array = np.empty([0, len(out_labels)])
in_array = np.empty([0, 4])

df_reduced = df_all[in_labels+out_labels]

# %%
# reduce resolution in f and pv space
coarsed_coordinates = np.linspace(0,1, 101) # factor 5 to original resolution
df_coarsed = df_reduced[df_all['f'].isin(coarsed_coordinates)]
df_coarsed = df_coarsed[df_coarsed['pv'].isin(coarsed_coordinates)]

def expand_data_4D(
    df_input, input_labels=in_labels, output_labels=out_labels, new_dim="4th"):

    out_array = np.empty([0, len(output_labels)])
    in_array = np.empty([0, 4])


    for i in range(1,points_new_dim+1):
        tmp = df_input[out_labels].values * np.exp(i*0.1)
        out_array = np.vstack([out_array, tmp])

        df_input[new_dim] = i*0.1
        tmp_in = df_input[input_labels + [new_dim]].values
        in_array = np.vstack([in_array, tmp_in])

    df_out = pd.DataFrame(out_array, columns=output_labels)
    df_in = pd.DataFrame(in_array, columns=input_labels + [new_dim])

    df_4d = pd.concat([df_in, df_out], axis=1)
    return df_4d


# def expand_data_5D(
#     df_input, input_labels=in_labels, output_labels=out_labels, new_dim="5th"):
#
#     out_array = np.empty([0, len(output_labels)])
#     in_array = np.empty([0, 5])
#
#     for i in range(1,points_new_dim+1):
#         #tmp1 = df_input[out_labels].values * np.exp(i*0.1)
#         tmp2 = df_input[out_labels].values * np.cos(i*0.1)
#         out_array = np.vstack([out_array, tmp2])
#
#         df_input[new_dim] = i*0.1
#         tmp_in = df_input[input_labels + [new_dim]].values
#         in_array = np.vstack([in_array, tmp_in])
#
#     df_out = pd.DataFrame(out_array, columns=output_labels)
#     df_in = pd.DataFrame(in_array, columns=input_labels + [new_dim])
#
#     df_5d = pd.concat([df_in, df_out], axis=1)
#     return df_5d


df_4d = expand_data_4D(df_input = df_coarsed)
# df_5d = expand_data_5D(df_input = df_4d,input_labels=in_labels.append('4th'))

# %%

# df_4d.reset_index(inplace=True)
# df_coarsed.reset_index(inplace=True)

df_4d.to_hdf('table_reduced_4D.h5',key='data',)
df_coarsed.to_hdf('table_reduced_3D.h5',key='data')

# # %%
# px.scatter_3d(
#     df_4d[(df_4d["4th"] == 1) & (df_4d.zeta == 0)].sample(25000),
#     x="f",
#     y="pv",
#     z="PVs",
# )

# %%
