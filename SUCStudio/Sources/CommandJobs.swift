import Foundation
import Combine

enum CommandJobTools {
    static let names=["run_command","command_status","wait_command","stop_command"]
    static let management=["command_status","wait_command","stop_command"]
    static func integer(_ value:String?,default fallback:Int,range:ClosedRange<Int>,name:String) throws -> Int {
        guard let value else{return fallback}
        guard let parsed=Int(value),range.contains(parsed) else{throw MCPConnection.failure("\(name): usa un numero intero tra \(range.lowerBound) e \(range.upperBound).")}
        return parsed
    }
}

struct CommandJobSnapshot {
    let id:UUID, state:String, elapsed:Double, exitCode:Int32?, output:String
    let outputOffset:Int, nextOffset:Int, omittedBytes:Int, totalOutputBytes:Int
    let failure:String?
    var running:Bool{state=="running"}
    var json:String {
        var value:[String:Any]=["job_id":id.uuidString,"state":state,"elapsed_seconds":(elapsed*10).rounded()/10,"output":output,"output_offset":outputOffset,"next_offset":nextOffset,"omitted_bytes":omittedBytes,"total_output_bytes":totalOutputBytes,"output_remaining_bytes":max(0,totalOutputBytes-nextOffset)]
        if let exitCode{value["exit_code"]=exitCode}
        if let failure{value["error"]=failure}
        if running {value["next_action"]="Use wait_command with this job_id; waiting does not stop or restart the process."}
        return String(decoding:(try? JSONSerialization.data(withJSONObject:value,options:[.sortedKeys])) ?? Data(),as:UTF8.self)
    }
}

/// Runtime-only process state. A bounded tail is retained; snapshots explicitly report omitted bytes.
final class CommandJob:Identifiable,@unchecked Sendable {
    let id=UUID(), sessionID:UUID, recordID:UUID, command:String
    let started=Date()
    private let startedUptime=ProcessInfo.processInfo.systemUptime
    private let lock=NSLock()
    private var state="running", output=Data(), totalBytes=0, finishedUptime:Double?, code:Int32?, failure:String?
    var task:Task<Void,Never>?
    init(sessionID:UUID,recordID:UUID,command:String){self.sessionID=sessionID;self.recordID=recordID;self.command=command}
    func append(_ data:Data){
        lock.lock();defer{lock.unlock()}
        totalBytes+=data.count;output.append(data)
        if output.count>256*1024 {output=Data(output.suffix(256*1024))}
    }
    func finish(_ state:String,exitCode:Int32?=nil,error:String?=nil){
        lock.lock();defer{lock.unlock()};self.state=state;code=exitCode;failure=error;finishedUptime=ProcessInfo.processInfo.systemUptime
    }
    func snapshot(offset:Int?=nil)->CommandJobSnapshot {
        lock.lock();defer{lock.unlock()}
        let base=totalBytes-output.count
        let requested=min(totalBytes,max(0,offset ?? max(0,totalBytes-64*1024)))
        var from=max(base,requested),end=min(totalBytes,max(base,requested)+64*1024)
        func byte(_ absolute:Int)->UInt8 {output[output.startIndex+absolute-base]}
        func continuation(_ value:UInt8)->Bool {value & 0xC0 == 0x80}
        // Cursors are byte offsets, but complete UTF-8 scalars cross read/tail boundaries.
        for _ in 0..<3 {if from<end && continuation(byte(from)){from+=1}else{break}}
        if end<totalBytes {
            for _ in 0..<3 {if end>from && continuation(byte(end)){end-=1}else{break}}
        } else if state=="running",end>from {
            var lead=end-1
            for _ in 0..<3 {if lead>from && continuation(byte(lead)){lead-=1}else{break}}
            let first=byte(lead),length=first>=0xC2 && first<=0xDF ? 2:first>=0xE0 && first<=0xEF ? 3:first>=0xF0 && first<=0xF4 ? 4:1
            if end-lead<length {end=lead}
        }
        let data=Data(output.dropFirst(from-base).prefix(end-from))
        return CommandJobSnapshot(id:id,state:state,elapsed:(finishedUptime ?? ProcessInfo.processInfo.systemUptime)-startedUptime,exitCode:code,output:String(decoding:data,as:UTF8.self),outputOffset:from,nextOffset:end,omittedBytes:max(0,from-requested)+(offset == nil ? from:0),totalOutputBytes:totalBytes,failure:failure)
    }
    var running:Bool{lock.lock();defer{lock.unlock()};return state=="running"}
}

@MainActor final class CommandJobs:ObservableObject {
    @Published private(set) var jobs:[CommandJob]=[]
    var onFinish:((CommandJob)->Void)?
    var onChange:(()->Void)?
    private var deliveredOffsets:[UUID:Int]=[:]
    private let helperURL:URL?
    init(helperURL:URL?=nil){self.helperURL=helperURL}
    func running(_ session:UUID?=nil)->Bool{jobs.contains{(session == nil || $0.sessionID==session) && $0.running}}
    func owned(_ id:String?,session:UUID)->CommandJob?{guard let id,let uuid=UUID(uuidString:id) else{return nil};return jobs.first{$0.id==uuid && $0.sessionID==session}}
    func lookup(_ id:String?,session:UUID) throws -> CommandJob {
        guard let job=owned(id,session:session) else{throw MCPConnection.failure("Comando sconosciuto in questa sessione. Nessun processo è stato avviato o riavviato.")};return job
    }
    func stop(_ id:UUID,session:UUID){jobs.first{$0.id==id && $0.sessionID==session}?.task?.cancel();onChange?()}
    func stopAll(_ session:UUID?=nil){for job in jobs where session == nil || job.sessionID==session {job.task?.cancel()};onChange?()}
    func shutdown() async {stopAll();for task in jobs.compactMap(\.task){await task.value}}
    func start(command:String,root:URL,session:UUID,recordID:UUID,timeout:Double?) throws -> CommandJob {
        guard !command.isEmpty,command.utf8.count<=32768 else{throw MCPConnection.failure("Comando assente o troppo grande.")}
        guard !running(session) else{
            let active=jobs.first{$0.sessionID==session && $0.running}!
            throw MCPConnection.failure("Un comando è già in esecuzione: \(active.id.uuidString). Usa wait_command o stop_command; non avviarne una copia.")
        }
        guard jobs.filter(\.running).count<8 else{throw MCPConnection.failure("Massimo 8 comandi attivi. Attendi o interrompi un comando.")}
        while jobs.count>=32,let index=jobs.firstIndex(where:{!$0.running}){deliveredOffsets.removeValue(forKey:jobs[index].id);jobs.remove(at:index)}
        let job=CommandJob(sessionID:session,recordID:recordID,command:command),runner=WorkspaceCommand(helperURL:helperURL)
        jobs.append(job);onChange?()
        job.task=Task { [weak self] in
            do {
                let result=try await runner.run(command,root:root,timeout:timeout,onOutput:{job.append($0)})
                let code=Int32(result.components(separatedBy:"\n")[0].replacingOccurrences(of:"Exit status: ",with:"")) ?? -1
                job.finish(code==0 ? "completed":"failed",exitCode:code)
            } catch is CancellationError {job.finish("cancelled")}
            catch let error as CommandTimeLimitError {job.finish("timed_out",error:error.localizedDescription)}
            catch {job.finish("failed",error:error.localizedDescription)}
            self?.objectWillChange.send();self?.onChange?();self?.onFinish?(job)
        }
        return job
    }
    func wait(_ job:CommandJob,seconds:Int) async throws {
        let end=ProcessInfo.processInfo.systemUptime+Double(seconds)
        while job.running && ProcessInfo.processInfo.systemUptime<end {
            try Task.checkCancellation();try await Task.sleep(nanoseconds:50_000_000)
        }
        try Task.checkCancellation()
    }
    func perform(_ action:WorkspaceAction,session:ChatSession,root:URL?,recordID:UUID) async throws -> String {
        try Task.checkCancellation()
        let offset:Int?=try action.arguments["output_offset"].map{try CommandJobTools.integer($0,default:0,range:0...Int.max,name:"output_offset")}
        let seconds=try CommandJobTools.integer(action.arguments["wait_seconds"],default:action.name=="wait_command" ? 30:1,range:0...60,name:"wait_seconds")
        let job:CommandJob
        if action.name=="run_command" {
            guard session.shellCommandsEnabled,let root else{throw MCPConnection.failure("Comandi shell disabilitati per questa sessione.")}
            let timeout:Double?=try action.arguments["timeout_seconds"].map{Double(try CommandJobTools.integer($0,default:0,range:1...86400,name:"timeout_seconds"))}
            job=try start(command:action.arguments["command"] ?? "",root:root,session:session.id,recordID:recordID,timeout:timeout)
        } else {job=try lookup(action.arguments["job_id"],session:session.id)}
        if action.name=="stop_command" {stop(job.id,session:session.id);try await wait(job,seconds:2)}
        else if action.name != "command_status" {
            do{try await wait(job,seconds:seconds)}catch{
                if action.name=="run_command" {job.task?.cancel()}
                throw error
            }
        }
        return read(job,offset:offset)
    }
    func read(_ job:CommandJob,offset:Int?=nil)->String {
        let value=job.snapshot(offset:offset ?? deliveredOffsets[job.id] ?? 0)
        deliveredOffsets[job.id]=value.nextOffset
        return value.json
    }
}
