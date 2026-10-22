import Foundation

/// Calls live only in this process. A wait never starts, retries or extends a call.
enum MCPJobTools {
    static let management=["mcp_status","mcp_wait","mcp_cancel"]
}
@MainActor final class MCPJob {
    let id=UUID(), sessionID:UUID, recordID:UUID, serverID:UUID
    let started=ProcessInfo.processInfo.systemUptime
    var finished:Double?, state="running", result:String?, error:String?, delivered=false
    var task:Task<Void,Never>?
    init(session:UUID,record:UUID,server:UUID){sessionID=session;recordID=record;serverID=server}
    var running:Bool{state=="running"}
    var json:String {
        var value:[String:Any]=["job_id":id.uuidString,"state":state,"elapsed_seconds":(((finished ?? ProcessInfo.processInfo.systemUptime)-started)*10).rounded()/10]
        if let result{value["result"]=result}
        if let error{value["error"]=error}
        if running{value["next_action"]="Use mcp_wait or mcp_status with this job_id. Do not repeat mcp_call; the original call is still running."}
        if state=="cancelled" || state=="timed_out" {value["outcome_uncertain"]=true;value["notice"]="Cancellation is best effort; already executed effects cannot be undone. Do not automatically repeat this call."}
        return String(decoding:(try? JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])) ?? Data(),as:UTF8.self)
    }
}
@MainActor final class MCPJobs {
    private(set) var jobs:[MCPJob]=[]
    var onFinish:((MCPJob)->Void)?
    func running(_ session:UUID?=nil)->Bool{jobs.contains{(session == nil || $0.sessionID==session) && $0.running}}
    func owned(_ id:String?,session:UUID)->MCPJob? {guard let id,let uuid=UUID(uuidString:id) else{return nil};return jobs.first{$0.id==uuid && $0.sessionID==session}}
    func stopAll(_ session:UUID?=nil){
        for job in jobs where session == nil || job.sessionID==session {
            // Stop ends the turn. Retain its record, but never deliver an old result to a later turn.
            job.delivered=true;job.task?.cancel()
        }
    }
    func stopServer(_ server:UUID){for job in jobs where job.serverID==server{job.task?.cancel()}}
    func start(client:MCPConnection,tool:MCPTool,input:[String:Any],session:UUID,record:UUID,timeout:Double?) throws -> MCPJob {
        guard !running(session) else{throw MCPConnection.failure("Una chiamata MCP è già in corso: usa mcp_status o mcp_wait con il job_id esistente, senza duplicarla.")}
        guard jobs.filter(\.running).count<8 else{throw MCPConnection.failure("Massimo 8 chiamate MCP attive. Attendi o interrompi una chiamata.")}
        while jobs.count>=32,let index=jobs.firstIndex(where:{!$0.running && $0.delivered}) {jobs.remove(at:index)}
        guard jobs.count<64 else{throw MCPConnection.failure("Troppi risultati MCP non ancora letti: consulta i job esistenti.")}
        let job=MCPJob(session:session,record:record,server:tool.serverID);jobs.append(job)
        job.task=Task { [weak self] in
            do {job.result=try await client.call(tool,arguments:input,timeout:timeout);try Task.checkCancellation();job.state="completed"}
            catch is CancellationError {job.state="cancelled";job.error="Chiamata interrotta; esito remoto eventualmente incerto."}
            catch {job.state=Task.isCancelled ? "cancelled":(error as NSError).domain=="SUCStudio.MCP" && (error as NSError).code==408 ? "timed_out":"failed";job.error=error.localizedDescription}
            job.finished=ProcessInfo.processInfo.systemUptime;self?.onFinish?(job)
        }
        return job
    }
    func wait(_ job:MCPJob,seconds:Int) async throws {
        let end=ProcessInfo.processInfo.systemUptime+Double(seconds)
        while job.running && ProcessInfo.processInfo.systemUptime<end {try Task.checkCancellation();try await Task.sleep(nanoseconds:50_000_000)}
        try Task.checkCancellation()
    }
    func read(_ job:MCPJob)->String {if !job.running{job.delivered=true};return job.json}
    func manage(_ action:WorkspaceAction,session:UUID) async throws -> String {
        guard let job=owned(action.arguments["job_id"],session:session) else{throw MCPConnection.failure("Chiamata MCP sconosciuta in questa sessione. Nessuno strumento è stato avviato o ripetuto.")}
        let seconds=try CommandJobTools.integer(action.arguments["wait_seconds"],default:30,range:0...60,name:"wait_seconds")
        if action.name=="mcp_cancel" {job.task?.cancel();try await wait(job,seconds:2)}
        else if action.name=="mcp_wait" {try await wait(job,seconds:seconds)}
        return read(job)
    }
}
extension StudioStore {
    func mcpJobFinished(_ job:MCPJob) {
        objectWillChange.send()
        mutateSession(job.sessionID){s in
            if let index=s.actions?.firstIndex(where:{$0.id==job.recordID && $0.state=="running"}) {
                s.actions?[index].state=job.state=="completed" ? "completed":job.state=="cancelled" ? "cancelled":"error"
                s.actions?[index].result=job.json
            }
        }
        operationJournal.update(id:job.recordID.uuidString,state:job.state=="completed" ? "Completata":job.state=="cancelled" ? "Interrotta":"Errore")
        if job.state=="failed" || job.state=="timed_out" {actionNotice(job.sessionID,L("Chiamata MCP terminata con errore. Apri il registro azioni per i dettagli."))}
        markSessionUnreadIfNeeded(job.sessionID);scheduleChatQueue(job.sessionID)
    }
}
