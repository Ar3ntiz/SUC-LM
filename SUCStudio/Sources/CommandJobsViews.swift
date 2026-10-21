import SwiftUI

struct CommandJobsView:View {
    @EnvironmentObject var store:StudioStore
    @ObservedObject var jobs:CommandJobs
    let sessionID:UUID
    @State private var expanded=false
    var body:some View {
        let items=jobs.jobs.filter{$0.sessionID==sessionID}
        if !items.isEmpty {
            DisclosureGroup(isExpanded:$expanded) {
                TimelineView(.periodic(from:.now,by:1)) { _ in
                    ScrollView {
                        VStack(alignment:.leading,spacing:8) {
                            ForEach(items.suffix(4)) {job in
                                let value=job.snapshot()
                                DisclosureGroup {
                                    Text(job.command).font(StudioTypography.ui(.caption,design:.monospaced)).textSelection(.enabled)
                                    if !value.output.isEmpty {Text(value.output).font(StudioTypography.ui(.caption2,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)}
                                    if value.omittedBytes>0 {Text(L("Anteprima: \(value.omittedBytes) byte precedenti non mostrati.")).font(StudioTypography.ui(.caption2)).foregroundStyle(.secondary)}
                                    if let failure=value.failure {Text(LT(failure)).font(StudioTypography.ui(.caption)).foregroundStyle(.red)}
                                    Text("job_id: "+job.id.uuidString).font(StudioTypography.ui(.caption2,design:.monospaced)).foregroundStyle(.tertiary).textSelection(.enabled)
                                } label: {
                                    HStack {
                                        Text(String(job.command.prefix(70))).font(StudioTypography.ui(.caption,design:.monospaced)).lineLimit(1)
                                        Spacer()
                                        Text(LT(["running":"In esecuzione","completed":"Completata","failed":"Errore","cancelled":"Interrotta","timed_out":"Durata massima raggiunta"][value.state] ?? value.state)).font(StudioTypography.ui(.caption2))
                                        Text(String(format:"%.0f s",value.elapsed)).font(StudioTypography.ui(.caption2)).monospacedDigit()
                                        if value.running {Button(L("Interrompi")){store.cancelSelected()}.buttonStyle(.borderless)}
                                    }
                                }
                            }
                        }.padding(.top,6)
                    }.frame(maxHeight:190)
                }
            } label: {
                HStack {
                    Label(L("Comandi della sessione"),systemImage:"terminal")
                    if jobs.running(sessionID) {ProgressView().controlSize(.mini);Spacer();Button(L("Interrompi comandi")){store.cancelSelected()}.buttonStyle(.borderless)}
                }.font(StudioTypography.ui(.caption))
            }.padding(.horizontal,18).padding(.vertical,8).background(Color.secondary.opacity(0.04)).accessibilityIdentifier("command-jobs")
        }
    }
}
