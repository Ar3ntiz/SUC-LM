import Foundation
import CryptoKit

/// A conservative runtime heuristic, not proof that an operation has no side effects.
/// Only consecutive identical terminal results count. Job observations never count.
@MainActor final class ActionLoopGuard {
    private struct Entry {let key:String;let record:UUID}
    private var turns:[UUID:[Entry]]=[:]
    func reset(_ session:UUID){turns.removeValue(forKey:session)}
    private func management(_ name:String)->Bool {CommandJobTools.management.contains(name) || MCPJobTools.management.contains(name)}
    private func key(_ action:WorkspaceAction)->String {
        var value=action
        if ["run_command","mcp_call"].contains(action.name) {value.arguments.removeValue(forKey:"wait_seconds");value.arguments.removeValue(forKey:"timeout_seconds")}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        return digest((try? encoder.encode(value)) ?? Data())
    }
    private func digest(_ data:Data)->String{SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
    private func result(_ record:ActionRecord)->String? {
        guard ["completed","error"].contains(record.state) else{return nil}
        if ["run_command","mcp_call"].contains(record.name),
           let object=try? JSONSerialization.jsonObject(with:Data(record.result.utf8)) as? [String:Any],object["job_id"] != nil {
            guard let state=object["state"] as? String,["completed","failed"].contains(state) else{return nil}
            // Job IDs, timings and cursors change on every invocation, not its outcome.
            let fields=record.name=="run_command" ? ["state","exit_code","output","total_output_bytes","omitted_bytes"]:["state","result","error"]
            let stable=object.filter{fields.contains($0.key)}
            guard let data=try? JSONSerialization.data(withJSONObject:stable,options:[.sortedKeys]) else{return nil}
            return digest(data)
        }
        return digest(Data(record.result.utf8))
    }
    func note(_ action:WorkspaceAction,session:UUID,recordID:UUID) {
        guard !management(action.name) else{return}
        turns[session,default:[]].append(Entry(key:key(action),record:recordID))
        if turns[session]!.count>100 {turns[session]!.removeFirst(turns[session]!.count-100)}
    }
    func blocks(_ action:WorkspaceAction,session:UUID,records:[ActionRecord])->Bool {
        guard !management(action.name),let recent=turns[session],recent.count>=3 else{return false}
        let target=key(action),last=Array(recent.suffix(3))
        guard last.allSatisfy({$0.key==target}) else{return false}
        let values=last.compactMap{entry in records.first(where:{$0.id==entry.record}).flatMap{result($0)}}
        return values.count==3 && Set(values).count==1
    }
}
