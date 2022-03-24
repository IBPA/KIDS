CARD Download README

ONTOLOGIES:

Data on the four core ontologies underlying the CARD are provided in three formats: OBO, 
CSV, and JSON. They are:

ARO - The Antibiotic Resistance Ontology that serves as the primary organizing principle 
of the CARD.

MO - CARD's Model Ontology which describes the different antimicrobial resistance gene 
detection models encoded in the CARD and used by the Resistance Gene Identifier.

NCBITaxon - A slim version of the NCBI Taxonomy used by the CARD to track source pathogen
of molecular sequences.

RO - CARD's Relationship Ontology which describes the different relationship types between 
ontology terms.

FASTA:

Nucleotide and corresponding protein FASTA downloads are available as separate files for 
each model type.  For example, the "protein homolog" model type contains sequences of
antimicrobial resistance genes that do not include mutation as a determinant of resistance
- these data are appropriate for BLAST analysis of metagenomic data or searches excluding 
secondary screening for resistance mutations. In contrast, the "protein variant" model 
includes reference wild type sequences used for mapping SNPs conferring antimicrobial 
resistance - without secondary mutation screening, analyses using these data will include 
false positives for antibiotic resistant gene variants or mutants.

MODELS:

The file "card.json" contains the complete data for all of CARD's AMR detection models, 
including reference sequences, SNP mapping data, model parameters, and ARO classification.

INDEX FILES:

The file "aro_index.csv" contains a list of ARO tagging of GenBank accessions stored in 
the CARD.

The file "aro_categories_index.csv" contains a list in which GenBank accessions stored 
in the CARD are cross-referenced with the major categories within the ARO determinant 
branch. These categories reflect both target drug class and mechanism of resistance, so 
CARD identifiers may have more than one cross-reference. For more complex categorization
of the data, use the ontology files described above.
