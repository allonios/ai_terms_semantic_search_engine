import rdflib

graph = rdflib.Graph()

graph.parse("../ignored/data/autoencoder/autoencoder2.owl")

query1 = """
SELECT ?a ?b
WHERE {
    ?a 
    untitled-ontology-4:hasInput
    <http://www.semanticweb.org/mohammed_almasry/ontologies/2021/5/untitled-ontology-4#feature_detectors_input> ;
    
    untitled-ontology-4:uses
    <http://www.semanticweb.org/mohammed_almasry/ontologies/2021/5/untitled-ontology-4#en_autoencoder> ;
    
    untitled-ontology-4:description ?b .
}
"""

query2 = """
SELECT ?a ?b
WHERE {
    ?a untitled-ontology-4:learns ?b .    
}
"""

query3 = """
SELECT ?a ?b
WHERE {
    ?a <http://www.semanticweb.org/mohammed_almasry/ontologies/2021/5/untitled-ontology-4#weight> ?b .    
}
"""

result = list(graph.query(query1))




with open("result.txt", "w") as file:
    file.write(
        "\n".join(
            map(
                lambda res: f"{res.a}\n...\n{res.b}\n",
                result
            )
        )
    )


