export class DataStore {
    dictionary: Record<string, any>
    yamlString: string

    constructor(data: Record<string, any>) {
        this.dictionary = data['dictionary']
        this.yamlString = data['yaml_string']
    }

    public filter(key: string, dictionary: Record<any, any> = null): Record<string, any> {
        if (dictionary == null) {
            dictionary = this.dictionary
        }

        if (key == '') {
            return dictionary
        }

        let query = new RegExp(key, 'g')

        let dict: Record<string, any> = {}
        for (let k in dictionary) {
            if (k.search(query) !== -1 || (typeof dictionary[k] == 'string' && dictionary[k].search(query) !== -1)) {
                dict[k] = dictionary[k]
            } else if (typeof dictionary[k] === 'object') {
                let res = this.filter(key, dictionary[k])
                if (Object.keys(res).length > 0) {
                    dict[k] = res
                }
            }
        }

        return dict
    }
}