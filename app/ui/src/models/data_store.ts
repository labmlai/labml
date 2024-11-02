export class DataStore {
    dictionary: Record<string, any>
    yamlString: string

    constructor(data: Record<string, any>) {
        this.dictionary = data['dictionary']
        this.yamlString = data['yaml_string']
    }
}