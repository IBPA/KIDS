require 'rest-client'
require '/home/jyoun/Jason/Opensource/ClassyFire/lib/classyfire_api.rb'

File.open('./ant_temp.tsv', 'r') do |antibiotics_file|
	antibiotics_file.each_line do |antibiotic|
		File.open('result.txt', 'a') do |result_file|
			submit_result = ClassyFireAPI.submit_query(label=antibiotic, antibiotic)
			query_id = JSON.parse(submit_result)['id']

			puts 'Processing ' + antibiotic.strip + ' (Query ID: ' + query_id.to_s + ')'

			query_result = ClassyFireAPI.get_query(query_id=query_id)
			entities = JSON.parse(query_result)['entities']

			result_taxonomy = ''
			if entities.length == 1
				# major
				kingdom_array = entities[0]['kingdom']
				kingdom_name = kingdom_array['name']
				result_taxonomy += kingdom_name

				superclass_array = entities[0]['superclass']
				superclass_name = superclass_array['name']
				result_taxonomy += ':' + superclass_name

				class_array = entities[0]['class']
				class_name = class_array['name']
				result_taxonomy += ':' + class_name

				subclass_array = entities[0]['subclass']
				if not subclass_array.nil?

					subclass_name = subclass_array['name']
					result_taxonomy += ':' + subclass_name

					# minor
					intermediate_nodes = entities[0]['intermediate_nodes']

					if intermediate_nodes.length != 0
						intermediate_nodes.each do |node|
							result_taxonomy += ':' + node['name']
						end

						direct_parent_array = entities[0]['direct_parent']
						direct_parent_name = direct_parent_array['name']
						result_taxonomy += ':' + direct_parent_name
					end

				end
			end
			puts "\t" + result_taxonomy
			result_file.write(antibiotic.strip + "\t" + result_taxonomy + "\n")
		end
	end
end
