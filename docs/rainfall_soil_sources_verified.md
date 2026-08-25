# Rainfall and Soil Moisture Source Verification — SIH 2026 PS26192

**Verification date:** 25 August 2026  
**Pilot area used for coverage checks:** Guwahati, Assam.

## Important correction

The two research reports are strong starting points, but several access/coverage details needed verification. The most important correction is the MOSDAC SMAP Soil Moisture/SWI product: its official metadata lists a bounding box of **5–24°N, 68–90°E**, which does **not** include Guwahati (approximately 26°N, 91.7°E). Therefore it should not be presented as a direct soil-moisture source for the Guwahati pilot without confirming a newer/expanded product. citeturn4search0

## Top 5 verification results

### GSMaP_ISRO Rain
- Type: Rainfall
- Status: **VERIFIED — MOSDAC SSO required**
- Official URL: https://www.mosdac.gov.in/gsmap-isro-rain

### NASA GPM IMERG
- Type: Rainfall
- Status: **VERIFIED — Earthdata/PPS login required for direct downloads**
- Official URL: https://gpm.nasa.gov/data/imerg

### IMD 0.25° Gridded Daily Rainfall
- Type: Rainfall
- Status: **VERIFIED — official public archive**
- Official URL: https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html

### ERA5-Land
- Type: Soil moisture / environmental
- Status: **VERIFIED — CDS login required**
- Official URL: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land

### NASA SMAP L4 SPL4SMGP
- Type: Soil moisture
- Status: **VERIFIED — free Earthdata login required**
- Official URL: https://nsidc.org/data/spl4smgp/versions/8

## Top 3 rainfall sources

1. **GSMaP_ISRO Rain — VERIFIED.** MOSDAC's official page confirms the dataset, 0.1° spatial resolution, hourly resolution, March 2000 onward, HDF5 distribution, and SSO-based download. It is gauge-adjusted using IMD information. citeturn16search0
2. **NASA GPM IMERG — VERIFIED.** NASA confirms 30-minute products, global coverage, V07B coverage from January 1998, and multiple download formats. Direct PPS downloads require free registration/Earthdata access. citeturn0search1turn0search2
3. **IMD 0.25° Gridded Daily Rainfall — VERIFIED.** The official IMD Climate Research & Services portal is live and exposes the 0.25° NetCDF archive. Recent peer-reviewed studies also document downloading this dataset directly from the IMD portal and cite the 1901–2024 archive. citeturn1search0turn15search0

## Top 2 soil-moisture sources

1. **ERA5-Land — VERIFIED.** Copernicus confirms a global 0.1°/9 km hourly dataset from January 1950 to present, with soil-water layers and CC-BY licensing. Requests require CDS login/registration. citeturn1search9turn1search2
2. **NASA SMAP L4 SPL4SMGP — VERIFIED.** NSIDC's current Version 8 page confirms 9 km, 3-hourly surface and root-zone soil moisture from 31 March 2015 to present, HDF5 format, and a free NASA Earthdata Login requirement. citeturn2search4

## Other important verified sources

- **CHIRPS v3:** official CHC page confirms 0.05°, 1981–near-present, 60°N–60°S, multiple time steps and public-domain/CC BY 4.0 status. citeturn6search0turn6search6
- **Copernicus CLMS Soil Water Index v4:** official CLMS page confirms global 0.1° daily data from 2007 to present and open/free access through CDSE. citeturn3search0
- **ISRO EOS-04 soil moisture:** official ISRO/NRSC documentation confirms an operational ~500 m surface soil-moisture product over India and public availability through Bhoonidhi. citeturn5search15turn5search16
- **ISMN:** official ISMN documentation confirms free registered access to in-situ soil-moisture data and station-dependent temporal sampling. citeturn17search0turn17search7

## Recommended prototype stack

### Rainfall
- **GSMaP_ISRO** — primary India-focused hourly rainfall.
- **IMERG Early** — high-frequency near-real-time cross-check.
- **IMD gridded rainfall** — long historical Indian baseline.
- **IMD/MOSDAC AWS/ARG** — local validation/bias correction.
- **CHIRPS v3** — optional high-resolution historical cross-validation.

### Soil moisture
- **ERA5-Land** — primary historical hourly soil-moisture/context feature.
- **SMAP L4** — satellite-assimilated 3-hourly surface/root-zone feature.
- **EOS-04 500 m** — high-resolution spatial anchor where coverage/product access is suitable.
- **ISMN** — ground validation where stations exist.

## Sample rainfall download recommendation

For the team's first real-file download, **CHIRPS v3** is the easiest source because the official CHC repository is publicly accessible without a login and provides GeoTIFF/NetCDF products. The repository explicitly documents HTTP/FTP/RSYNC distribution. citeturn6search0turn6search3

For a strictly India-government source, use the **IMD 0.25° NetCDF archive**; recent research confirms that the archive is publicly downloadable. citeturn15search0turn15search8

## Important access interpretation

**VERIFIED** means the official dataset/page exists and its provider documentation confirms the product.  
**VERIFIED — LOGIN REQUIRED** means the dataset is real and accessible, but direct download requires a free account. This is not the same as "paid."  
**PAID** was not found among the main recommended sources.

