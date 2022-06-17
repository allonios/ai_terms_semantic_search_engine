from owlready2 import get_ontology, default_world

autoencoder = get_ontology(
    "../ignored/data/autoencoder/autoencoder2.owl"
).load()

base_iri = autoencoder.base_iri

query = f"""
PREFIX onto:   <{base_iri}>
Select ?b
{{
    onto:Rnn_autoencoder onto:description ?b .
}}
"""


query_res = list(default_world.sparql(query))
