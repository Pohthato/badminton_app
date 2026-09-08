ALTER TABLE `analysisSessions` ADD `failureReason` text;
--> statement-breakpoint
ALTER TABLE `analysisSessions` ADD `lastWorkerStatusAt` timestamp;
