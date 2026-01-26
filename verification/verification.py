import csv

########################################
# FILES
########################################

source_csv = 'ALL_SOURCES_drug_disease_merged.csv'  # File containing source interactions
interaction_tsv = 'disease_negative_samples.tsv'   # File with interactions to verify
output_tsv = 'interactions_validated_with_sources.tsv'  # Output file

########################################
# LOAD SOURCE DATA
########################################

def load_source_data(file_path):
    """
    Load data from the source CSV file into a dictionary.
    Key: (drug_name_lower, disease_name_lower)
    Value: dictionary containing all row information
    """
    source_dict = {}
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            drug_name = row['drug_name'].strip()
            disease_name = row['disease_name'].strip()
            
            # Create a normalized key (lowercase)
            key = (drug_name.lower(), disease_name.lower())
            
            # Store all information from the row
            source_dict[key] = {
                'drug_name': drug_name,
                'disease_name': disease_name,
                'source': row.get('source', '').strip(),
                'internal_source': row.get('internal_source', '').strip(),
                'drug_id': row.get('drug_id', '').strip(),
                'disease_id': row.get('disease_id', '').strip(),
                'extra_metadata': row.get('extra_metadata', '').strip()
            }
    
    print(f"Loaded {len(source_dict)} interactions from {file_path}")
    return source_dict

########################################
# INTERACTION VERIFICATION
########################################

def check_interactions(interaction_file, source_dict):
    """
    Check whether the interactions in the TSV file exist in the source data
    """
    results = []
    
    with open(interaction_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split by tab
            parts = line.split('\t')
            if len(parts) == 2:
                disease, drug = parts
                disease = disease.strip()
                drug = drug.strip()
                
                # Create lookup key (lowercase)
                search_key = (drug.lower(), disease.lower())
                
                # Check if interaction exists
                if search_key in source_dict:
                    source_data = source_dict[search_key]
                    results.append({
                        'disease': disease,
                        'drug': drug,
                        'interaction': 'yes',
                        'source': source_data['source'],
                        'internal_source': source_data['internal_source'],
                        'drug_name_in_source': source_data['drug_name'],
                        'disease_name_in_source': source_data['disease_name'],
                        'drug_id': source_data['drug_id'],
                        'disease_id': source_data['disease_id'],
                        'extra_metadata': source_data['extra_metadata']
                    })
                else:
                    results.append({
                        'disease': disease,
                        'drug': drug,
                        'interaction': 'no',
                        'source': 'none',
                        'internal_source': 'none',
                        'drug_name_in_source': '',
                        'disease_name_in_source': '',
                        'drug_id': '',
                        'disease_id': '',
                        'extra_metadata': ''
                    })
            else:
                print(f"Malformed line ignored: {line}")
    
    return results

########################################
# WRITE RESULTS
########################################

def write_results_to_tsv(results, output_file):
    """
    Write detailed results to a TSV file
    """
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = [
            'Disease_Input', 'Drug_Input',
            'Interaction_Found', 'Source', 'Internal_Source',
            'Drug_Name_In_Source', 'Disease_Name_In_Source',
            'Drug_ID', 'Disease_ID', 'Extra_Metadata'
        ]
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'Disease_Input': result['disease'],
                'Drug_Input': result['drug'],
                'Interaction_Found': result['interaction'],
                'Source': result['source'],
                'Internal_Source': result['internal_source'],
                'Drug_Name_In_Source': result['drug_name_in_source'],
                'Disease_Name_In_Source': result['disease_name_in_source'],
                'Drug_ID': result['drug_id'],
                'Disease_ID': result['disease_id'],
                'Extra_Metadata': result['extra_metadata']
            })
    
    print(f"Results written to {output_file}")

########################################
# SIMPLIFIED VERSION (LIMITED COLUMNS)
########################################

def write_simple_results_to_tsv(results, output_file):
    """
    Simplified version with limited columns
    """
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['Disease', 'Drug', 'Interaction', 'Source', 'Found_In']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        for result in results:
            interaction = result['interaction']
            source = result['source']
            internal_source = result['internal_source']
            
            found_in = ""
            if interaction == 'yes':
                if source == 'AACT' and internal_source == 'ClinicalTrials.gov':
                    found_in = 'source.csv (ClinicalTrials.gov)'
                elif source:
                    found_in = f'source.csv ({source})'
                else:
                    found_in = 'source.csv'
            else:
                found_in = 'Not found'
            
            writer.writerow({
                'Disease': result['disease'],
                'Drug': result['drug'],
                'Interaction': interaction,
                'Source': source if source != 'none' else '',
                'Found_In': found_in
            })
    
    print(f"Simplified results written to {output_file}")

########################################
# ANALYSIS AND STATISTICS
########################################

def analyze_results(results):
    """
    Display statistics about the results
    """
    total = len(results)
    found = sum(1 for r in results if r['interaction'] == 'yes')
    not_found = total - found
    
    print("\n" + "=" * 50)
    print("RESULT STATISTICS")
    print("=" * 50)
    print(f"Total interactions checked: {total}")
    print(f"Interactions found: {found} ({found / total * 100:.1f}%)")
    print(f"Interactions not found: {not_found} ({not_found / total * 100:.1f}%)")
    
    if found > 0:
        print("\nExamples of found interactions:")
        for result in results[:3]:
            if result['interaction'] == 'yes':
                print(f"  {result['drug']} - {result['disease']}: {result['source']}")
    
    if not_found > 0:
        print("\nExamples of interactions not found:")
        count = 0
        for result in results:
            if result['interaction'] == 'no' and count < 3:
                print(f"  {result['drug']} - {result['disease']}")
                count += 1

########################################
# MAIN EXECUTION
########################################

def main():
    print("STARTING INTERACTION VERIFICATION")
    print("=" * 50)
    
    print("Loading source data...")
    source_data = load_source_data(source_csv)
    
    print("Checking interactions...")
    results = check_interactions(interaction_tsv, source_data)
    
    write_results_to_tsv(results, output_tsv)
    
    simple_output = 'interactions_validated_simple.tsv'
    write_simple_results_to_tsv(results, simple_output)
    
    analyze_results(results)
    
    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETED")
    print("Generated files:")
    print(f"  - {output_tsv} (detailed version)")
    print(f"  - {simple_output} (simplified version)")
    print("=" * 50)

########################################
# RUN
########################################

if __name__ == "__main__":
    main()
