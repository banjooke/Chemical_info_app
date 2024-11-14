from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import streamlit as st

#Initialize OpenAI API KEY

openai_api_key = st.secrets["openai"]["OPENAI_API_KEY"]

#set up Langchain components
llm = ChatOpenAI(
    model = "gpt-4",
    api_key = openai_api_key,
    max_tokens=6000)

#Define prompt template

template = """
Provide detailed technical information about {chemical}. You include the physical propperties, chemical properties, common uses, safety protocols and handling requirement
"""
prompt = PromptTemplate(input_variable = ["chemical"], template=template)

#Initialize LLM Chain
chain = LLMChain(llm = llm, prompt=prompt)

#List of Chemicals

chemicals = [
    "Acetone", "Ammonia", "Benzene", "Chlorine", "Ethanol", "Formaldehyde", "Hydrochloric Acid",
    "Hydrogen Peroxide", "Methanol", "Nitric Acid", "Phenol", "Sodium Hydroxide", "Sulfuric Acid",
    "Toluene", "Xylene", "Acetic Acid", "Butane", "Propane", "Ethylene", "Propylene", "Styrene",
    "Acetylene", "Isopropanol", "Glycerol", "Urea", "Polyethylene", "Polypropylene", "PVC",
    "Polyester", "Nylon", "Acrylic Acid", "Caprolactam", "Aniline", "Epoxy Resin", "Silicone",
    "Chloroform", "Isocyanates", "Bromine", "Ethylene Glycol", "Diethylene Glycol", "Phosphoric Acid",
    "Carbon Disulfide", "Carbon Tetrachloride", "Pyridine", "Cyclohexane", "Cyclopentane", "Naphthalene",
    "Thiourea", "Citric Acid", "Calcium Carbonate", "Magnesium Sulfate", "Potassium Hydroxide",
    "Sodium Bicarbonate", "Tetrahydrofuran", "Hexane", "Pentane", "Octane", "Nonane", "Decane",
    "Methane", "Ethane", "Butadiene", "Maleic Anhydride", "Fumaric Acid", "Succinic Acid",
    "Sodium Hypochlorite", "Potassium Permanganate", "Sodium Chloride", "Calcium Chloride",
    "Potassium Chloride", "Barium Sulfate", "Lithium Hydroxide", "Lithium Carbonate", "Borax",
    "Perchloric Acid", "Chlorobenzene", "Iodine", "Zinc Chloride", "Copper Sulfate", "Silver Nitrate",
    "Nickel Sulfate", "Ferrous Sulfate", "Aluminum Chloride", "Tungsten Carbide", "Titanium Dioxide",
    "Silicon Dioxide", "Magnesium Oxide", "Calcium Oxide", "Barium Hydroxide", "Cesium Hydroxide",
    "Neodymium Oxide", "Gallium Nitrate", "Indium Phosphide", "Arsenic", "Phosphine", "Silane",
    "Hydrogen", "Oxygen", "Nitrogen", "Carbon Dioxide", "Carbon Monoxide", "Helium", "Neon",
    "Argon", "Krypton", "Xenon", "Radon", "Sulfur Hexafluoride", "Nitrous Oxide", "Ammonium Nitrate",
    "Calcium Hypochlorite", "Potassium Nitrate", "Sodium Nitrate", "Magnesium Chloride",
    "Aluminum Oxide", "Calcium Sulfate", "Barium Carbonate", "Sodium Sulfate", "Potassium Sulfate",
    "Ammonium Sulfate", "Calcium Nitrate", "Ferric Chloride", "Cupric Chloride", "Zinc Oxide",
    "Cobalt Chloride", "Manganese Sulfate", "Lead Nitrate", "Mercuric Chloride", "Cadmium Sulfate",
    "Chromium Oxide", "Vanadium Pentoxide", "Molybdenum Trioxide", "Tungsten Trioxide", "Boron Trifluoride",
    "Magnesium Oxide", "Beryllium Oxide", "Thallium Nitrate", "Potassium Bicarbonate", "Lithium Bromide",
    "Aluminum Sulfate", "Alum", "Sodium Metabisulfite", "Potassium Dichromate", "Chromic Acid",
    "Sodium Phosphate", "Phosphoric Acid", "Dichloromethane", "1,2-Dichloroethane", "1,4-Dioxane",
    "Trichloroethylene", "Tetrachloroethylene", "Hexachlorobutadiene", "2,4-Dinitrotoluene",
    "1,3,5-Trinitrobenzene", "Trinitrotoluene", "Hydrogen Cyanide", "Methyl Isocyanate", 
    "Ethylene Oxide", "Propylene Oxide", "Acrolein", "Acrylonitrile", "Benzyl Chloride", 
    "Phenylhydrazine", "Hydrazine", "Nitrobenzene", "Nitromethane", "Nitroethane", "Diethyl Ether",
    "Tetraethyl Lead", "Tetramethyl Lead", "Selenium Dioxide", "Tellurium Dioxide", "Tin Tetrachloride",
    "Antimony Trioxide", "Bismuth Oxide", "Lead Oxide", "Cobalt Oxide", "Nickel Oxide", 
    "Iron Oxide", "Cupric Oxide", "Vanadium Oxide", "Boron Nitride", "Gallium Arsenide", 
    "Silicon Carbide", "Zirconium Dioxide", "Hafnium Dioxide", "Yttrium Oxide", "Erbium Oxide",
    "Dysprosium Oxide", "Holmium Oxide", "Thulium Oxide", "Ytterbium Oxide", "Lutetium Oxide",
    "Tungsten Oxide", "Tantalum Oxide", "Niobium Pentoxide", "Chromium Trioxide", "Zinc Sulfide",
    "Cadmium Sulfide", "Lead Sulfide", "Arsenic Trioxide", "Sodium Chromate", "Potassium Chromate",
    "Sodium Molybdate", "Ammonium Molybdate", "Calcium Fluoride", "Magnesium Fluoride", "Lithium Fluoride",
    "Barium Fluoride", "Potassium Fluoride", "Sodium Fluoride", "Ammonium Bifluoride", "Hydrofluoric Acid",
    "Fluorosilicic Acid", "Titanium Tetrachloride", "Tin Chloride", "Zirconium Tetrachloride",
    "Iron Chloride", "Copper Chloride", "Sodium Dichromate", "Potassium Dichromate", "Chromium Nitrate",
    "Vanadium Nitrate", "Magnesium Phosphate", "Ammonium Phosphate", "Calcium Phosphate",
    "Iron Phosphate", "Barium Phosphate", "Zinc Phosphate", "Manganese Phosphate", "Nickel Phosphate",
    "Cerium Oxide", "Lanthanum Oxide", "Praseodymium Oxide", "Gadolinium Oxide", "Samarium Oxide",
    "Europium Oxide", "Terbium Oxide", "Yttrium Fluoride", "Neodymium Fluoride", "Samarium Fluoride",
    "Erbium Fluoride", "Magnesium Bromide", "Barium Bromide", "Zinc Bromide", "Copper Bromide", 
    "Manganese Bromide", "Cadmium Bromide", "Ammonium Bromide", "Ammonium Iodide", "Sodium Iodide",
    "Potassium Iodide", "Magnesium Iodide", "Calcium Iodide", "Lithium Iodide", "Copper Iodide"
]

#Strealit App Layout
st.title("Chemical Info Helper")
chemical = st.selectbox("Select a chemical:", chemicals)

if st.button("Get Information"):
    #Generate response from the LLM
    with st.spinner("Retrieving information..."):
        response = chain.run({"chemical": chemical})
    st.subheader(f"Technical Information for {chemical}")
    st.write(response)