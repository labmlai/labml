import {RunListItemModel} from "../models/run_list"
import {RunStatuses} from "../models/status"

export function getSearchQuery() {
    return localStorage.getItem('searchQuery') || ''
}

export function setSearchQuery(query: string) {
    localStorage.setItem('searchQuery', query)
}

export function runsFilter(run: RunListItemModel, searchText: string) {
    setSearchQuery(searchText)

    let {tags, query, mainTags} = extractTags(searchText)
    tags = tags.concat(mainTags)

    if (tags.length == 0 && query == "") {
        return true
    }

    const queryRegex = new RegExp(query.toLowerCase(), 'g')
    const tagRegex: RegExp[] = []
    let hasRunningTag = false
    for (let tag of tags) {
        tagRegex.push(new RegExp(`(^|\\s)${tag.toLowerCase()}(?=\\s|$)`, 'g'))
        if (tag.toLowerCase() == 'running') {
            hasRunningTag = true
        }
    }

    let matchName = query == "" || run.name.toLowerCase().search(queryRegex) !== -1
    let matchComment = query == "" || run.comment.toLowerCase().search(queryRegex) !== -1
    let matchTags = tags.length == 0 || tagRegex.every(tag => run.tags.join(' ').toLowerCase().search(tag) !== -1)

    if (!matchTags)
        return false
    if (hasRunningTag && run.run_status.status != 'in progress') {
        return false
    }

    return matchName || matchComment
}

export function extractTags(input: string): { tags: string[], query: string, mainTags: string[] } {
    const tags: string[] = []
    const regex = /:(\S+)/g;
    let match: RegExpExecArray | null

    while ((match = regex.exec(input)) !== null) {
        tags.push(match[0].substring(1));
    }

    const mainTags: string[] = []
    const mainRegex = /\$(\S+)/g;
    let mainMatch: RegExpExecArray | null

    while ((mainMatch = mainRegex.exec(input)) !== null) {
        mainTags.push(mainMatch[0].substring(1));
    }

    const rest = input.replace(regex, '').replace(mainRegex, '').trim()

    return {tags: tags, query: rest, mainTags: mainTags};
}