"""The extra stocks only the Oracle observer asks about (docs/oracle-provider-task.md, section 5.2).

The list is the output of `scripts/select_oracle_extra.py`, second run of 2026-11-19, made at
2026-11-19T15:16:42+00:00 (the first run let two leveraged ETFs through; see the script's docstring). The rule
is written in the document BEFORE the list was used and it looks at no return of any kind: pool by
exchange, symbol and name; the 300 most liquid by the median dollar volume of the last 60 sessions
(SIP, daily); a stock is kept only if its 15-minute bars on the observer's own feed are at least as
complete as those of the 10th-percentile stock already followed (the bar of that run: 0.9962).

**This list is never edited by hand.** Another list is another widening: it needs its own entry in the
document before it runs. A test pins the count and the shape. A name that is also in the research
universe or the watchlist is left out at use (`data.universe.oracle_extra_symbols`).
"""

MADE_AT = "2026-11-19T15:16:42+00:00"
COMPLETENESS_BAR = 0.9962

ORACLE_EXTRA: tuple[str, ...] = tuple(
    (
        "A AAOI ABNB ADM ADSK AEM AFRM AGNC AJG AKAM ALAB ALB ALL ALNY ANET APA APH APLD APO "
        "APP ARES ASST ASTS ASX ATI AU AWK AXON AXTI B BBY BDX BE BKR BMNR BNY BP BURL BX CAH "
        "CARR CBRS CCJ CCL CDE CDNS CDW CEG CF CFG CHRW CIEN CIFR CLS CLX CME CMG CMI CMS CNC "
        "CNP CNQ COHR COR CPRT CRCL CRH CRL CRS CRWD CRWV CTSH CVNA DAL DASH DDOG DECK DELL DG "
        "DHI DINO DKNG DLTR DOCN DOV DOW DT DVN DXCM EAT EBAY ECHO ECL ED EIX EL ENTG EQT ETR "
        "ETSY EW EXE EXPD EXPE FAST FERG FFIV FIG FIS FISV FITB FIX FLUT FN FOXA FPS FSLR FTAI "
        "FTNT GEHC GEV GH GIS GLW GOOG GWW HAL HALO HBAN HCA HIG HIMS HL HLT HONA HOOD HPE HPQ "
        "HSY HUM HUT HWM IBKR ICE ILMN INSM IONQ IQV IR JBL JCI JHX KDP KEY KEYS KHC KKR KLAC "
        "KMB KNX KR LEN LITE LNG LPLA LULU LUV LVS LYV MCHP MCK MDB MDLN MELI MKSI MLM MNST MOD "
        "MP MPWR MRNA MRVL MSI MTSI MTZ NDAQ NET NI NLY NOK NRG NTAP NTRA NU NUE NVO NVS NXPI O "
        "OKE OKLO OKTA ON ONDS ONON ONTO OTIS P PANW PBR PCAR PCG PEG PINS PPL PWR QSR QXO RBLX "
        "RBRK RCL RDDT RF RIOT RIVN RKLB RKT RL ROK ROP ROST RVMD RVTY S SMCI SMR SMTC SN SNDK "
        "SNOW SNPS SPCX SPOT SRE STM STRL SU SUNB SWKS SYF SYY TDG TEAM TEL TEM TER TLN TOST "
        "TPR TRGP TRV TSCO TSM TT TTD TTMI TTWO TWLO TWST TXG TYL U UAL UBER UMC URI USFD VALE "
        "VEEV VICI VIK VMRK VRSN VRT VST VTR WAT WCN WDAY WDC WEC WELL WPM WULF XEL XYL XYZ YUM "
        "ZM ZS"
    ).split()
)
