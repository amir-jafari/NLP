#%% --------------------------------------------------------------------------------------------------------------------
# pip install deeplake
# The below code must run deeplake 4.0, not deeplake 3.0. The function in two different versions
# changed a lot.
#%% --------------------------------------------------------------------------------------------------------------------
import deeplake
import numpy as np
import os

#%% --------------------------------------------------------------------------------------------------------------------
deeplake.convert(
    src='al://org_name/v3_dataset',
    dst='al://org_name/v4_dataset'
)
ds = deeplake.open('deeplake_local')
print("Migration complete!")
print(f"v4 dataset length: {len(ds)}")