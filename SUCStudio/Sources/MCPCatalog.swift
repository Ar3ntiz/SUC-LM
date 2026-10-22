import Foundation

/// Discovery stays small even when connected servers publish hundreds of large schemas.
enum MCPCatalog {
    static let actions = ["mcp_list", "mcp_schema"]
    static let pageSize = 10
    static let fragmentLimit = 16 * 1024
    private static func normalized(_ name:String)->String {
        name.lowercased().split(whereSeparator:{!$0.isLetter && !$0.isNumber}).joined(separator:" ")
    }
    private static func alias(_ name:String)->String {
        var tokens=normalized(name).split(separator:" ").map(String.init)
        if tokens.last=="mcp" {tokens.removeLast()}
        return tokens.joined(separator:" ")
    }
    /// Resolve identity from the currently connected catalog, never a fuzzy prefix.
    /// An ambiguous name/alias must be qualified with its configured name or UUID.
    static func serverID(_ requested:String,tools:[MCPTool]) throws->UUID {
        let servers=Dictionary(grouping:tools,by:{$0.serverID})
        let name=requested.trimmingCharacters(in:.whitespacesAndNewlines)
        if let id=UUID(uuidString:name),servers[id] != nil {return id}
        let exact=servers.filter{$0.value.first?.serverName.trimmingCharacters(in:.whitespacesAndNewlines).lowercased()==name.lowercased()}.map(\.key)
        let matches:[UUID]
        if !exact.isEmpty {matches=exact}
        else if name.isEmpty {matches=Array(servers.keys)}
        else {let key=alias(name);matches=key.isEmpty ? []:servers.filter{alias($0.value.first?.serverName ?? "")==key}.map(\.key)}
        if matches.count==1 {return matches[0]}
        let available=servers.sorted{$0.key.uuidString<$1.key.uuidString}.map{($0.value.first?.serverName ?? "MCP")+" ["+$0.key.uuidString+"]"}.joined(separator:", ")
        if servers.isEmpty {throw MCPConnection.failure("Nessuno strumento MCP disponibile: verifica connessioni e catalogo nei Settings.")}
        if matches.isEmpty {throw MCPConnection.failure("Nome server MCP non riconosciuto: "+name+". Server disponibili: "+available)}
        throw MCPConnection.failure("Nome server MCP ambiguo: "+name+". Specifica il nome esatto o UUID. Server disponibili: "+available)
    }
    static func prompt(_ tools:[MCPTool]) -> String {
        guard !tools.isEmpty else{return ""}
        let servers=Dictionary(grouping:tools,by:{$0.serverName}).sorted{$0.key<$1.key}
            .map{"- \($0.key): \($0.value.count) strumenti"}.joined(separator:"\n")
        return """
        Server MCP connessi:
        \(servers)
        Gli schemi sono caricati su richiesta per non saturare il contesto.
        Per cercare strumenti usa suc_action mcp_list con arguments server, query (facoltativa), offset (facoltativo, stringa numerica). I risultati sono paginati.
        Usa il nome del server indicato nel catalogo. Mai dedurre che è scollegato da un nome non riconosciuto: leggi i server disponibili nell’errore. Sono ammessi UUID e alias univoci senza il suffisso MCP.
        Prima di chiamare uno strumento usa suc_action mcp_schema con arguments server e tool. Gli schemi grandi sono consultabili per frammenti: path (JSON Pointer) e offset permettono di leggere tutti i campi senza troncamenti silenziosi.
        Per eseguire usa suc_action mcp_call con arguments server, tool, input (oggetto JSON codificato come stringa), wait_seconds facoltativo (0–60, default 1) e timeout_seconds facoltativo (durata totale 1–86400 secondi, nessuna scadenza imposta se omesso). Il risultato contiene job_id e state. Se running, usa mcp_wait(job_id, wait_seconds 0–60, default 30), mcp_status(job_id) o mcp_cancel(job_id). L’attesa non interrompe né riavvia lo strumento: non ripetere mcp_call. Stato/attesa/annullamento del proprio job non richiedono nuova approvazione né consumano il limite azioni. L’annullamento è best effort e non annulla effetti già eseguiti; non ripetere automaticamente una chiamata dall’esito incerto. I job non sopravvivono al riavvio. Attendi sempre il risultato reale. Le chiamate richiedono le approvazioni della sessione; ricerca catalogo e lettura schema non eseguono strumenti.
        Per richiedere un’azione rispondi con un solo oggetto JSON: {"suc_action":{"name":"mcp_list","arguments":{"server":"NOME_SERVER","query":"termine"}}}.
        """
    }
    private static func encode(_ value:[String:Any]) throws -> String {
        let data=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])
        guard data.count<=32*1024 else{throw MCPConnection.failure("Metadati dello schema troppo grandi: richiedi un path più specifico.")}
        return String(decoding:data,as:UTF8.self)
    }
    private static func offset(_ action:WorkspaceAction) throws -> Int {
        guard let value=Int(action.arguments["offset"] ?? "0"),value>=0 else{throw MCPConnection.failure("offset deve essere un intero non negativo.")}
        return value
    }
    static func read(_ action:WorkspaceAction,tools:[MCPTool]) throws -> String {
        let server=action.arguments["server"] ?? ""
        let id:UUID? = server.isEmpty && action.name=="mcp_list" ? nil:try serverID(server,tools:tools)
        let available=tools.filter{id == nil || $0.serverID==id}
        guard !available.isEmpty else{throw MCPConnection.failure("Server MCP non connesso o privo di strumenti.")}
        let start=try offset(action)
        if action.name=="mcp_list" {
            let terms=(action.arguments["query"] ?? "").lowercased().split(whereSeparator:{$0.isWhitespace}).map(String.init)
            let matches=available.filter{tool in let text=(tool.name+" "+tool.description).lowercased();return terms.allSatisfy{text.contains($0)}}.sorted{($0.serverName,$0.name)<($1.serverName,$1.name)}
            let page=matches.dropFirst(min(start,matches.count)).prefix(pageSize)
            var result:[String:Any]=["total":matches.count,"offset":start,"tools":page.map{["server":$0.serverName,"tool":$0.name,"description":String($0.description.prefix(300))]}]
            if start<matches.count,page.count<matches.count-start {result["next_offset"]=start+page.count}
            return try encode(result)
        }
        guard action.name=="mcp_schema",let tool=available.first(where:{$0.name==action.arguments["tool"]}) else{throw MCPConnection.failure("Strumento MCP non trovato: usa mcp_list.")}
        let path=action.arguments["path"] ?? ""
        var value:Any=try JSONSerialization.jsonObject(with:Data(tool.schema.utf8))
        if !path.isEmpty {
            guard path.hasPrefix("/") else{throw MCPConnection.failure("path deve essere un JSON Pointer, per esempio /properties.")}
            for token in path.dropFirst().components(separatedBy:"/") {
                let key=token.replacingOccurrences(of:"~1",with:"/").replacingOccurrences(of:"~0",with:"~")
                if let object=value as? [String:Any],let child=object[key] {value=child}
                else if let array=value as? [Any],let index=Int(key),array.indices.contains(index) {value=array[index]}
                else{throw MCPConnection.failure("Percorso nello schema MCP non trovato.")}
            }
        }
        let encoded=try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.fragmentsAllowed])
        if encoded.count<=fragmentLimit {return try encode(["server":tool.serverName,"tool":tool.name,"path":path,"schema":value])}
        func childPath(_ key:String)->String {path+"/"+key.replacingOccurrences(of:"~",with:"~0").replacingOccurrences(of:"/",with:"~1")}
        var result:[String:Any]=["server":tool.serverName,"tool":tool.name,"path":path,"bytes":encoded.count,"fragmented":true,"offset":start,"instruction":"Richiedi un path indicato per leggere il frammento completo; usa next_offset per gli altri campi."]
        let entries:[(String,Any)]
        if let object=value as? [String:Any] {entries=object.sorted{$0.key<$1.key}.map{($0.key,$0.value)}}
        else if let array=value as? [Any] {entries=array.enumerated().map{(String($0.offset),$0.element)}}
        else {throw MCPConnection.failure("Valore scalare nello schema oltre 16 KiB: consultabile integralmente nel pannello Strumenti MCP.")}
        let page=entries.dropFirst(min(start,entries.count)).prefix(pageSize)
        result["total"]=entries.count
        result["entries"]=page.map{key,child -> [String:Any] in
            var entry:[String:Any]=["key":key,"path":childPath(key)]
            if let object=child as? [String:Any] {entry["kind"]="object";entry["keys"]=Array(object.keys.sorted().prefix(20));entry["key_count"]=object.count;if let type=object["type"] {entry["type"]=type};if let description=object["description"] as? String {entry["description_preview"]=String(description.prefix(200))}}
            else if let array=child as? [Any] {entry["kind"]="array";entry["count"]=array.count}
            else if let string=child as? String {entry["kind"]="string";entry["preview"]=String(string.prefix(200));entry["characters"]=string.count}
            else {entry["value"]=child}
            return entry
        }
        if start<entries.count,page.count<entries.count-start {result["next_offset"]=start+page.count}
        return try encode(result)
    }
}
