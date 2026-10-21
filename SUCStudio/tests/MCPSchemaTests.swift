import Foundation
@main struct MCPSchemaTests {
    @MainActor static func main() async throws {
        let reader=LineReader(maxBytes:32),expected=["alpha","beta","gamma"]
        var received=[String]()
        for chunk in ["al","pha\nbeta\ngam","ma\n"] {reader.consume(Data(chunk.utf8)){received.append(String(decoding:$0,as:UTF8.self))}}
        precondition(received==expected)
        let reaID=UUID(),otherID=UUID()
        let rea=MCPTool(serverID:reaID,serverName:"REA mcp",name:"echo",description:"Fixture only",schema:"{\"type\":\"object\"}")
        for name in ["rea","REA"," REA MCP ","rea-mcp",reaID.uuidString.lowercased()] {
            let resolved=try MCPCatalog.serverID(name,tools:[rea]);precondition(resolved==reaID)
            let page=try MCPCatalog.read(WorkspaceAction(name:"mcp_list",arguments:["server":name]),tools:[rea]);precondition(page.contains("REA mcp"))
            let schema=try MCPCatalog.read(WorkspaceAction(name:"mcp_schema",arguments:["server":name,"tool":"echo"]),tools:[rea]);precondition(schema.contains("object"))
        }
        let other=MCPTool(serverID:otherID,serverName:"REA-MCP",name:"echo",description:"",schema:"{}")
        do{_=try MCPCatalog.serverID("rea",tools:[rea,other]);preconditionFailure("Ambiguous alias accepted")}catch{precondition(error.localizedDescription.contains("ambiguo"))}
        let exact=try MCPCatalog.serverID("REA mcp",tools:[rea,other]);precondition(exact==reaID)
        do{_=try MCPCatalog.serverID("missing",tools:[rea]);preconditionFailure("Unknown name accepted")}catch{precondition(error.localizedDescription.contains("disponibili") && !error.localizedDescription.contains("non connesso"))}
        do{_=try MCPCatalog.serverID("rea",tools:[]);preconditionFailure("Disconnected server accepted")}catch{}
        print("MCP server identity PASS: canonical names, unique aliases, UUID, ambiguity rejection and accurate diagnostics.")
        let fixture=CommandLine.arguments[1]
        for transport in ["stdio","http"] {
            for oversized in [false,true] {
                let flag=oversized ? "--oversized-schema":"--large-schema"
                var config=MCPConfiguration(name:"Schema fixture",transport:transport,executable:"/usr/bin/python3",arguments:[fixture,flag],enabled:true)
                var process:Process?
                if transport=="http" {
                    let server=Process(),output=Pipe();server.executableURL=URL(fileURLWithPath:"/usr/bin/python3");server.arguments=[fixture,"--http",flag];server.standardOutput=output
                    try server.run();process=server
                    let data=output.fileHandleForReading.availableData
                    let port=Int(String(decoding:data,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines))!
                    config.endpoint="http://127.0.0.1:\(port)/mcp";config.headers=["Authorization":"Bearer fixture"]
                }
                defer{process?.terminate()}
                let client=MCPConnection(config,credentials:[:]);defer{client.stop()}
                do {
                    let tools=try await client.start()
                    precondition(!oversized && tools.count==1)
                    let schema=tools[0].schema
                    precondition(schema.utf8.count>1024*1024 && schema.utf8.count<MCPConnection.maxSchemaBytes)
                    precondition(!schema.contains("\n"))
                    let decoded=try JSONSerialization.jsonObject(with:Data(schema.utf8)) as! [String:Any]
                    let properties=decoded["properties"] as! [String:Any]
                    let payload=properties["payload"] as! [String:Any]
                    precondition((payload["description"] as! String).count==2*1024*1024)
                } catch {
                    guard oversized else{throw error}
                    precondition(error.localizedDescription.contains("large_schema") && error.localizedDescription.contains("3 MiB"))
                }
            }
        }
        for transport in ["stdio","http"] {
            var config=MCPConfiguration(name:"Result fixture",transport:transport,executable:"/usr/bin/python3",arguments:[fixture],enabled:true)
            var process:Process?
            if transport=="http" {
                let server=Process(),output=Pipe();server.executableURL=URL(fileURLWithPath:"/usr/bin/python3");server.arguments=[fixture,"--http"];server.standardOutput=output
                try server.run();process=server
                let port=String(decoding:output.fileHandleForReading.availableData,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines)
                config.endpoint="http://127.0.0.1:\(port)/mcp";config.headers=["Authorization":"Bearer fixture"]
            }
            defer{process?.terminate()}
            let client=MCPConnection(config,credentials:[:]);defer{client.stop()}
            let tools=try await client.start();let tool=tools.first{$0.name=="echo"}!
            // ASCII, Unicode and worst-case JSON escaping, including the exact boundary.
            for unit in ["x","é","\u{0001}"] {
                let result=try await client.call(tool,arguments:["result_bytes":MCPConnection.maxResultBytes,"unit":unit],timeout:120)
                precondition(result.utf8.count==MCPConnection.maxResultBytes)
                precondition(result==String(repeating:unit,count:MCPConnection.maxResultBytes/unit.utf8.count))
            }
            do {
                _=try await client.call(tool,arguments:["result_bytes":MCPConnection.maxResultBytes+1],timeout:120)
                preconditionFailure("Oversized MCP result accepted")
            } catch {precondition(error.localizedDescription.contains("oltre 4 MiB"))}
            print("MCP results \(transport): intact 4 MiB ASCII/Unicode/escaped text accepted; 4 MiB + 1 rejected.")
        }
        let serverID=UUID()
        let tools=(0..<27).map{MCPTool(serverID:serverID,serverName:"fixture",name:String(format:"tool_%02d",$0),description:"Fixture searchable",schema:"{\"type\":\"object\"}")}
        let first=try MCPCatalog.read(WorkspaceAction(name:"mcp_list",arguments:["server":"fixture"]),tools:tools)
        let page=try JSONSerialization.jsonObject(with:Data(first.utf8)) as! [String:Any]
        precondition(page["next_offset"] as? Int==10 && (page["tools"] as? [[String:Any]])?.count==10)
        let largeSchema:[String:Any]=["type":"object","properties":["large":["type":"string","description":String(repeating:"x",count:20_000)],"a/b~c":["type":"integer"]]]
        let largeTool=MCPTool(serverID:serverID,serverName:"fixture",name:"large",description:"",schema:String(decoding:try JSONSerialization.data(withJSONObject:largeSchema),as:UTF8.self))
        let fragment=try MCPCatalog.read(WorkspaceAction(name:"mcp_schema",arguments:["server":"fixture","tool":"large"]),tools:[largeTool])
        precondition(fragment.utf8.count<4096 && fragment.contains("fragmented"))
        let leaf=try MCPCatalog.read(WorkspaceAction(name:"mcp_schema",arguments:["server":"fixture","tool":"large","path":"/properties/a~1b~0c"]),tools:[largeTool])
        precondition(leaf.contains("integer"))
        precondition(MCPCatalog.prompt(tools+[largeTool]).utf8.count<4096)
        if let executable=ProcessInfo.processInfo.environment["SUC_TEST_REA_EXECUTABLE"] {
            let config=MCPConfiguration(name:"REA catalog check",transport:"stdio",executable:executable,arguments:["mcp"],enabled:true)
            let directory=URL(fileURLWithPath:executable).deletingLastPathComponent().path
            let client=MCPConnection(config,credentials:["PATH":directory+":/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"])
            defer{client.stop()}
            let tools=try await client.start()
            precondition(tools.contains{$0.name=="compare_web_captures"})
            precondition(tools.contains{$0.schema.utf8.count>128*1024})
            precondition(MCPCatalog.prompt(tools).utf8.count<4096)
            print("Actual REA catalog: \(tools.count) tools connected; largest compact schema \(tools.map{$0.schema.utf8.count}.max() ?? 0) bytes; no tools called.")
        }
        print("MCP schema tests PASS: stdio/HTTP accept intact compact schemas above 1 MiB, reject schemas above 3 MiB with tool-specific diagnostics.")
    }
}
