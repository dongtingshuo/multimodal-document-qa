#import <Cocoa/Cocoa.h>
#import <signal.h>

@interface DocQAAppDelegate : NSObject <NSApplicationDelegate, NSWindowDelegate>
@property(nonatomic, strong) NSTask *supervisor;
@property(nonatomic, copy) NSString *statusPath;
@property(nonatomic, strong) NSTimer *statusTimer;
@property(nonatomic, strong) NSButton *retryButton;
@property(nonatomic, strong) NSButton *openButton;
@property(nonatomic, copy) NSString *uiURL;
@property(nonatomic, assign) BOOL stopping;
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) NSTextField *statusLabel;
@property(nonatomic, copy) NSString *projectDirectory;
@end

@implementation DocQAAppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)notification {
    NSApp.activationPolicy = NSApplicationActivationPolicyRegular;
    self.projectDirectory = [[[NSBundle mainBundle] bundlePath] stringByDeletingLastPathComponent];
    [self installMenu];
    [self installWindow];
    [self startService];
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    if (self.stopping) return NSTerminateLater;
    self.stopping = YES;
    [self.statusTimer invalidate];
    self.statusLabel.stringValue = @"正在关闭服务，请稍候…";
    if (!self.supervisor.isRunning) {
        if (self.statusPath) [[NSFileManager defaultManager] removeItemAtPath:self.statusPath error:NULL];
        return NSTerminateNow;
    }
    [self.supervisor terminate];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        [self.supervisor waitUntilExit];
        dispatch_async(dispatch_get_main_queue(), ^{
            [[NSFileManager defaultManager] removeItemAtPath:self.statusPath error:NULL];
            [NSApp replyToApplicationShouldTerminate:YES];
        });
    });
    return NSTerminateLater;
}

- (void)windowWillClose:(NSNotification *)notification {
    if (!self.stopping) {
        [NSApp terminate:nil];
    }
}

- (void)quit:(id)sender {
    [NSApp terminate:nil];
}

- (void)installMenu {
    NSMenu *mainMenu = [[NSMenu alloc] initWithTitle:@"文档问答系统"];
    NSMenuItem *appItem = [[NSMenuItem alloc] initWithTitle:@"文档问答系统" action:nil keyEquivalent:@""];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:@"文档问答系统"];
    [appMenu addItemWithTitle:@"退出文档问答系统" action:@selector(quit:) keyEquivalent:@"q"].target = self;
    appItem.submenu = appMenu;
    [mainMenu addItem:appItem];
    NSApp.mainMenu = mainMenu;
}

- (void)installWindow {
    NSRect frame = NSMakeRect(0, 0, 520, 240);
    self.window = [[NSWindow alloc] initWithContentRect:frame
                                               styleMask:(NSWindowStyleMaskTitled |
                                                          NSWindowStyleMaskClosable |
                                                          NSWindowStyleMaskMiniaturizable)
                                                 backing:NSBackingStoreBuffered
                                                   defer:NO];
    self.window.title = @"文档问答系统";
    self.window.delegate = self;
    self.window.releasedWhenClosed = NO;
    [self.window center];

    NSView *content = [[NSView alloc] initWithFrame:frame];
    self.statusLabel = [[NSTextField alloc] initWithFrame:NSMakeRect(28, 128, 464, 82)];
    self.statusLabel.editable = NO;
    self.statusLabel.bordered = NO;
    self.statusLabel.drawsBackground = NO;
    self.statusLabel.font = [NSFont systemFontOfSize:15];
    self.statusLabel.alignment = NSTextAlignmentCenter;
    self.statusLabel.stringValue = @"正在启动服务……";
    [content addSubview:self.statusLabel];

    NSButton *button = [[NSButton alloc] initWithFrame:NSMakeRect(170, 20, 180, 34)];
    button.title = @"退出并关闭服务";
    button.bezelStyle = NSBezelStyleRounded;
    button.target = self;
    button.action = @selector(quit:);
    [content addSubview:button];
    NSArray *titles = @[@"打开问答页面", @"重新启动", @"查看启动日志"];
    SEL actions[] = {@selector(openPage:), @selector(retry:), @selector(openLog:)};
    for (NSUInteger i = 0; i < titles.count; i++) {
        NSButton *action = [[NSButton alloc] initWithFrame:NSMakeRect(28 + 158*i, 75, 148, 32)];
        action.title = titles[i];
        action.bezelStyle = NSBezelStyleRounded;
        action.target = self;
        action.action = actions[i];
        [content addSubview:action];
        if (i == 0) self.openButton = action;
        if (i == 1) self.retryButton = action;
    }
    self.window.contentView = content;
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)openPage:(id)sender {
    if (self.uiURL) [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:self.uiURL]];
}

- (void)openLog:(id)sender {
    NSString *path = [self.projectDirectory stringByAppendingPathComponent:@"data/app.log"];
    [[NSWorkspace sharedWorkspace] openURL:[NSURL fileURLWithPath:path]];
}

- (void)retry:(id)sender {
    if (!self.supervisor.isRunning && !self.stopping) [self startService];
}

- (void)refreshStatus:(NSTimer *)timer {
    NSData *data = [NSData dataWithContentsOfFile:self.statusPath];
    if (!data) return;
    id status = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
    if (![status isKindOfClass:[NSDictionary class]]) return;
    if (!self.supervisor.isRunning && ![status[@"state"] isEqual:@"error"]
            && ![status[@"state"] isEqual:@"stopped"]) {
        self.uiURL = nil;
        self.openButton.enabled = NO;
        self.statusLabel.stringValue = @"服务未能保持运行，可查看启动日志后重新启动。";
        return;
    }
    if ([status[@"message"] isKindOfClass:[NSString class]]) {
        self.statusLabel.stringValue = status[@"message"];
    }
    if ([status[@"state"] isEqual:@"ready"] && [status[@"ui_url"] isKindOfClass:[NSString class]]) {
        if (self.supervisor.isRunning) {
            self.uiURL = status[@"ui_url"];
            self.openButton.enabled = YES;
        } else {
            self.uiURL = nil;
            self.statusLabel.stringValue = @"服务已退出，可查看启动日志后重新启动。";
        }
    }
}

- (void)startService {
    [self.statusTimer invalidate];
    if (self.statusPath) [[NSFileManager defaultManager] removeItemAtPath:self.statusPath error:NULL];
    self.statusPath = [NSTemporaryDirectory() stringByAppendingPathComponent:
                      [NSString stringWithFormat:@"docqa-%@.json", NSUUID.UUID.UUIDString]];
    self.uiURL = nil;
    self.openButton.enabled = NO;
    self.retryButton.enabled = NO;
    self.statusLabel.stringValue = @"正在启动文档问答服务…";
    NSTask *task = [[NSTask alloc] init];
    task.executableURL = [NSURL fileURLWithPath:@"/bin/bash"];
    task.arguments = @[@"scripts/start.sh", @"--owner-pid",
                      [NSString stringWithFormat:@"%d", getpid()], @"--status-file", self.statusPath];
    task.currentDirectoryURL = [NSURL fileURLWithPath:self.projectDirectory];
    NSMutableDictionary *environment = [[[NSProcessInfo processInfo] environment] mutableCopy];
    environment[@"DOCQA_OPEN_BROWSER"] = @"true";
    task.environment = environment;

    NSString *logPath = [self.projectDirectory stringByAppendingPathComponent:@"data/app.log"];
    NSError *directoryError = nil;
    [[NSFileManager defaultManager] createDirectoryAtPath:[logPath stringByDeletingLastPathComponent]
                            withIntermediateDirectories:YES attributes:nil error:&directoryError];
    if (directoryError) {
        self.statusLabel.stringValue = @"无法创建启动日志，请检查项目目录权限。";
        self.retryButton.enabled = YES;
        return;
    }
    if (![[NSFileManager defaultManager] fileExistsAtPath:logPath]) {
        [[NSFileManager defaultManager] createFileAtPath:logPath contents:nil attributes:nil];
    }
    NSFileHandle *logHandle = [NSFileHandle fileHandleForWritingAtPath:logPath];
    if (!logHandle) {
        self.statusLabel.stringValue = @"无法写入启动日志，请检查项目目录权限。";
        self.retryButton.enabled = YES;
        return;
    }
    [logHandle seekToEndOfFile];
    [logHandle writeData:[[NSString stringWithFormat:@"\n--- %@ ---\n", NSDate.date]
                         dataUsingEncoding:NSUTF8StringEncoding]];
    task.standardOutput = logHandle;
    task.standardError = logHandle;
    task.terminationHandler = ^(NSTask *terminatedTask) {
        dispatch_async(dispatch_get_main_queue(), ^{
            if (self.stopping) {
                return;
            }
            [self.statusTimer invalidate];
            self.statusLabel.stringValue = terminatedTask.terminationStatus == 0
                ? @"服务已关闭，可点击重新启动。" : @"启动失败，可查看启动日志后重新启动。";
            [self refreshStatus:nil];
            self.retryButton.enabled = YES;
            self.openButton.enabled = NO;
        });
    };
    self.supervisor = task;
    @try {
        [task launch];
        self.statusTimer = [NSTimer scheduledTimerWithTimeInterval:0.5 target:self
                           selector:@selector(refreshStatus:) userInfo:nil repeats:YES];
    } @catch (NSException *exception) {
        self.statusLabel.stringValue = [NSString stringWithFormat:@"服务启动失败：%@", exception.reason];
        self.retryButton.enabled = YES;
    }
}

@end

int main(void) {
    @autoreleasepool {
        NSApplication *application = [NSApplication sharedApplication];
        DocQAAppDelegate *delegate = [[DocQAAppDelegate alloc] init];
        application.delegate = delegate;
        [application run];
    }
    return 0;
}
