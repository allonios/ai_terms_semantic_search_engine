import os

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from ontology_unifier.ontology_alignment import align_ontologies


class UnifyOntologyMixin:
    def save_file_to_storage(self, file):
        fs = FileSystemStorage()
        return fs.save(
            os.path.join(settings.UNIFIED_ONTOLOGIES_ROOT, file.name), file
        )

    def unify_file(self, file):
        final_file_path1 = os.path.join(
            settings.UNIFIED_ONTOLOGIES_ROOT, "final.owl"
        )
        final_file_path2 = os.path.join(
            settings.UNIFIED_ONTOLOGIES_ROOT, "final1.owl"
        )

        align_ontologies(final_file_path1, file, final_file_path2)

        os.remove(file)

        return settings.UNIFIED_ONTOLOGIES_URL + "final1.owl"
