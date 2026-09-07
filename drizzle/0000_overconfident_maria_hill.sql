CREATE TABLE `analysisSessions` (
	`id` varchar(28) NOT NULL,
	`userId` int NOT NULL,
	`sourceName` varchar(255) NOT NULL,
	`sourceStorageKey` varchar(768) NOT NULL,
	`sourceDurationMs` int,
	`selectedPlayer` enum('near','far') NOT NULL,
	`status` enum('draft','queued','processing','completed','failed') NOT NULL DEFAULT 'draft',
	`calibration` json NOT NULL,
	`workerJobId` varchar(128),
	`result` json,
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `analysisSessions_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `users` (
	`id` int AUTO_INCREMENT NOT NULL,
	`openId` varchar(64) NOT NULL,
	`name` text,
	`email` varchar(320),
	`loginMethod` varchar(64),
	`role` enum('user','admin') NOT NULL DEFAULT 'user',
	`createdAt` timestamp NOT NULL DEFAULT (now()),
	`updatedAt` timestamp NOT NULL DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	`lastSignedIn` timestamp NOT NULL DEFAULT (now()),
	CONSTRAINT `users_id` PRIMARY KEY(`id`),
	CONSTRAINT `users_openId_unique` UNIQUE(`openId`)
);
--> statement-breakpoint
ALTER TABLE `analysisSessions` ADD CONSTRAINT `analysisSessions_userId_users_id_fk` FOREIGN KEY (`userId`) REFERENCES `users`(`id`) ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX `analysisSessions_user_created_idx` ON `analysisSessions` (`userId`,`createdAt`);--> statement-breakpoint
CREATE INDEX `analysisSessions_worker_job_idx` ON `analysisSessions` (`workerJobId`);