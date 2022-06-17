from owlready2 import get_ontology, default_world
from owlready2 import Thing

autoencoder = get_ontology(
    "../ignored/data/autoencoder/autoencoder2.owl"
).load()


base_iri = autoencoder.base_iri


classes_query = """
SELECT ?x {
    ?x a owl:Class   
}
"""

# what is rnn_autoencoder input
## rnn_autoencoder
### disc...

learn_query = f"""
SELECT ?a
{{
    # ?a a owl:NamedIndividual .
    ?a ?b "language translator" .
}}
"""

learn_query1 = """
SELECT ?a ?b
{
    ?a rdf:about ?b .
}
"""

bilal_query = f"""
PREFIX onto:   {base_iri}
SELECT ?b
{{
    onto:Rnn_autoencoder onto:description ?b .
}}
"""

classes_query_res = list(default_world.sparql(classes_query))
learn_query_res = list(default_world.sparql(learn_query))
bilal_query_res = list(default_world.sparql(bilal_query))

search = autoencoder.search(iri=f"{base_iri}Rnn_autoencoder")




